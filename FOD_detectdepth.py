#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOD_detectdepth.py  (v5)
Deteksi FOD (YOLOv8n) + pengukuran jarak stereo (ELP dual-lens).
Model lensa dipilih OTOMATIS dari file kalibrasi:
  - dist 5 koefisien (k1,k2,p1,p2,k3)  -> model pinhole  (cv2.stereoRectify)
  - dist 4 koefisien + is_fisheye=True  -> model fisheye  (cv2.fisheye.*)

Stereo triangulasi langsung memberi koordinat 3D objek di frame kamera:
    Z = f * B / d,   X = (u - cx) * Z / f,   Y = (v - cy) * Z / f
Jarak 3D  D = sqrt(X^2 + Y^2 + Z^2)  tidak bergantung pada H maupun tilt.

Opsional (--ground): H, tilt, dan jarak horizontal di tanah diestimasi otomatis
dari bidang lantai/runway (RANSAC) pada peta disparitas.

Konvensi frame kamera: X kanan, Y bawah, Z depan. Satuan output: cm.

Jalankan:
  python FOD_detectdepth.py --weights best.pt --calib stereo_calib_fisheye2.npz
  python FOD_detectdepth.py --ground --log hasil.csv
  python FOD_detectdepth.py --model pinhole      (paksa model lensa)
Tombol: q = keluar | g = on/off estimasi bidang tanah | s = simpan frame
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================ KONFIGURASI ============================
WEIGHTS_PATH = r"D:\SKRIPSI_PROGRAM\Training_YOLOv8n\FOD_detection\weights\best.pt"
CALIB_PATH = r"D:\SKRIPSI_PROGRAM\stereo_calib_fisheye2.npz"

CAM_INDEX = 0
FRAME_W, FRAME_H = 2560, 720          # side-by-side, 1280x720 per mata
CROP_FRAC = 0.60                      # pakai ~60% area tengah

# Koreksi skala kedalaman (1.0 = tanpa koreksi)
DEPTH_SCALE = 1.0

# Stereo matching (SGBM)
DISP_SCALE = 0.5                      # 1.0 = resolusi penuh; 0.5 cocok Jetson Nano
NUM_DISP = 16 * 8
BLOCK = 7
MIN_VALID_FRAC = 0.15
INNER_FRAC = 0.5
MAX_RANGE_CM = 600.0

# Estimasi bidang tanah (RANSAC)
GROUND_ROW_START = 0.40
GROUND_STRIDE = 6
RANSAC_ITERS = 150
RANSAC_THRESH_CM = 2.0
GROUND_MIN_INLIER_FRAC = 0.30
GROUND_EMA = 0.2
# =====================================================================


def valid_P(P1, P2):
    return bool(np.isfinite(P1).all() and np.isfinite(P2).all()
                and P1[0, 0] > 10 and P2[0, 0] > 10)


def load_calib(path, model_opt="auto"):
    if not os.path.isfile(path):
        raise SystemExit(f"File kalibrasi tidak ditemukan: {path}")
    d = np.load(path, allow_pickle=True)
    print("[INFO] file kalibrasi:", path)
    print("[INFO] key:", d.files)

    for k in ("mtx_left", "dist_left", "mtx_right", "dist_right", "R", "T"):
        if k not in d.files:
            raise SystemExit(f"Key '{k}' tidak ada di file kalibrasi")

    K1 = d["mtx_left"].astype(np.float64)
    K2 = d["mtx_right"].astype(np.float64)
    D1 = np.asarray(d["dist_left"], np.float64).reshape(-1, 1)
    D2 = np.asarray(d["dist_right"], np.float64).reshape(-1, 1)
    R = d["R"].astype(np.float64)
    T = d["T"].astype(np.float64).reshape(3, 1)

    for nama, M in [("K1", K1), ("K2", K2), ("D1", D1), ("D2", D2), ("R", R), ("T", T)]:
        if not np.isfinite(M).all():
            raise SystemExit(f"{nama} mengandung NaN/inf -> kalibrasi rusak, kalibrasi ulang")
    if K1[0, 0] < 10 or K2[0, 0] < 10:
        raise SystemExit(f"Focal length K tidak valid: {K1[0, 0]}, {K2[0, 0]}")

    # ---- pilih model lensa ----
    flag_fish = ("is_fisheye" in d.files) and bool(d["is_fisheye"])
    if model_opt == "auto":
        if D1.size == 4 and flag_fish:
            model = "fisheye"
        else:
            model = "pinhole"
    else:
        model = model_opt
    if model == "fisheye" and (D1.size != 4 or D2.size != 4):
        raise SystemExit(f"Model fisheye butuh 4 koefisien distorsi, file berisi {D1.size}. "
                         "Gunakan --model pinhole atau kalibrasi ulang.")
    if model == "pinhole" and D1.size not in (4, 5, 8, 12, 14):
        raise SystemExit(f"Jumlah koefisien distorsi {D1.size} tidak didukung model pinhole")

    if "img_shape" in d.files:
        print("[INFO] img_shape:", d["img_shape"], "(harus = ukuran satu mata:",
              (FRAME_W // 2, FRAME_H), ")")

    saved = None
    if all(k in d.files for k in ("R1", "R2", "P1", "P2")):
        saved = (d["R1"].astype(np.float64), d["R2"].astype(np.float64),
                 d["P1"].astype(np.float64), d["P2"].astype(np.float64))
        if not valid_P(saved[2], saved[3]):
            print(f"[PERINGATAN] P1/P2 di .npz tidak valid (f = {saved[2][0, 0]:.1f} px) "
                  "-> rektifikasi dihitung ulang dari K, D, R, T")
            saved = None

    print(f"[INFO] model lensa: {model} | f_kiri = {K1[0, 0]:.1f} px | D1 = {D1.ravel()} | "
          f"T = {T.ravel()}")
    return K1, D1, K2, D2, R, T, saved, model


def to_cm_factor(T):
    """Deteksi satuan T dari panjang baseline (ELP ~6 cm): m / cm / mm -> cm."""
    b = float(np.linalg.norm(T))
    if b < 1.0:
        return 100.0   # meter
    if b < 20.0:
        return 1.0     # cm
    return 0.1         # mm


class StereoRig:
    def __init__(self, calib_path, model_opt="auto"):
        K1, D1, K2, D2, R, T, saved, model = load_calib(calib_path, model_opt)
        self.model = model
        self.unit = to_cm_factor(T)
        T = T * self.unit
        self.baseline = float(np.linalg.norm(T))          # cm
        self.K1, self.D1, self.K2, self.D2, self.R, self.T = K1, D1, K2, D2, R, T
        self.saved = saved
        self.size = (FRAME_W // 2, FRAME_H)
        self._using_saved = False
        self.build_maps()

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISP, blockSize=BLOCK,
            P1=8 * BLOCK * BLOCK, P2=32 * BLOCK * BLOCK,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    # ---------- peta rektifikasi ----------
    def _make_maps(self, R1, R2, P1, P2):
        if self.model == "fisheye":
            m1 = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, R1, P1, self.size, cv2.CV_16SC2)
            m2 = cv2.fisheye.initUndistortRectifyMap(self.K2, self.D2, R2, P2, self.size, cv2.CV_16SC2)
        else:
            m1 = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, self.size, cv2.CV_16SC2)
            m2 = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, self.size, cv2.CV_16SC2)
        return m1, m2

    def _valid_ratio(self, maps):
        """Fraksi piksel area crop yang benar-benar terisi gambar."""
        w, h = self.size
        ones = np.full((h, w), 255, np.uint8)
        out = cv2.remap(ones, maps[0], maps[1], cv2.INTER_LINEAR)
        x0, y0 = int(w * (1 - CROP_FRAC) / 2), int(h * (1 - CROP_FRAC) / 2)
        return float((out[y0:h - y0, x0:w - x0] > 250).mean())

    def _recompute_candidates(self):
        size = self.size
        cands = []
        if self.model == "fisheye":
            for bal, fov in [(0.0, 1.0), (0.3, 1.0), (0.0, 0.8), (0.6, 1.0), (1.0, 1.0)]:
                try:
                    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
                        self.K1, self.D1, self.K2, self.D2, size, self.R, self.T,
                        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=size,
                        balance=bal, fov_scale=fov)
                    cands.append((f"hitung ulang fisheye (balance={bal}, fov={fov})", R1, R2, P1, P2))
                except cv2.error as e:
                    print(f"[INFO] fisheye.stereoRectify balance={bal} fov={fov} gagal: {e}")
            # Cadangan: rotasi rektifikasi tidak bergantung pada distorsi, jadi ambil R1/R2
            # dari stereoRectify tanpa distorsi, lalu bangun P secara manual.
            try:
                zero = np.zeros((5, 1), np.float64)
                R1, R2, _, _, _, _, _ = cv2.stereoRectify(
                    self.K1, zero, self.K2, zero, size, self.R, self.T,
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
                w, h = size
                for fs in (1.0, 0.8, 0.6):
                    f = float(self.K1[0, 0]) * fs
                    P1 = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0]], np.float64)
                    P2 = P1.copy()
                    P2[0, 3] = -f * self.baseline
                    cands.append((f"manual fisheye (R dari stereoRectify, f={fs}*K)", R1, R2, P1, P2))
            except cv2.error as e:
                print(f"[INFO] kandidat manual fisheye gagal: {e}")
        else:
            for alpha in [0, 0.3, 1, -1]:
                try:
                    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
                        self.K1, self.D1, self.K2, self.D2, size, self.R, self.T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
                    cands.append((f"hitung ulang pinhole (alpha={alpha})", R1, R2, P1, P2))
                except cv2.error as e:
                    print(f"[INFO] stereoRectify alpha={alpha} gagal: {e}")
        return cands

    def build_maps(self, skip_saved=False):
        size = self.size
        cands = []
        if self.saved is not None and not skip_saved:
            R1, R2, P1, P2 = self.saved
            cands.append(("tersimpan di .npz", R1, R2, P1, P2))
        cands += self._recompute_candidates()

        best, best_ratio = None, -1.0
        for name, R1, R2, P1, P2 in cands:
            if not valid_P(P1, P2):
                print(f"[CEK] {name}: ditolak (f tidak valid)")
                continue
            m1, m2 = self._make_maps(R1, R2, P1, P2)
            ratio = min(self._valid_ratio(m1), self._valid_ratio(m2))
            print(f"[CEK] {name}: f = {P1[0, 0]:.1f} px, area terisi = {ratio * 100:.1f}%")
            if ratio > best_ratio:
                best, best_ratio = (name, m1, m2, P1), ratio
            if ratio >= 0.98:
                break
        if best is None:
            raise SystemExit("Tidak ada rektifikasi yang valid. Cek K, D, R, T dan --model "
                             "(pinhole/fisheye) di file kalibrasi.")
        if best_ratio < 0.9:
            print(f"[PERINGATAN] area terisi hanya {best_ratio * 100:.1f}% -> cek model lensa "
                  f"(--model), resolusi kamera = {FRAME_W}x{FRAME_H}, dan file kalibrasi")

        name, self.map1, self.map2, P1 = best
        self._using_saved = name.startswith("tersimpan")
        w, h = size
        self.x0 = int(w * (1 - CROP_FRAC) / 2)
        self.y0 = int(h * (1 - CROP_FRAC) / 2)
        self.x1, self.y1 = w - self.x0, h - self.y0
        self.f = float(P1[0, 0])
        self.cx = float(P1[0, 2]) - self.x0
        self.cy = float(P1[1, 2]) - self.y0
        print(f"[INFO] rektifikasi dipakai: {name} | baseline = {self.baseline:.2f} cm | "
              f"f = {self.f:.1f} px | crop = {self.x1 - self.x0}x{self.y1 - self.y0}")

    def maps_saved_in_use(self):
        return self._using_saved

    def rectify(self, frame):
        left = frame[:, :FRAME_W // 2]
        right = frame[:, FRAME_W // 2:]
        l = cv2.remap(left, self.map1[0], self.map1[1], cv2.INTER_LINEAR)
        r = cv2.remap(right, self.map2[0], self.map2[1], cv2.INTER_LINEAR)
        l = l[self.y0:self.y1, self.x0:self.x1]
        r = r[self.y0:self.y1, self.x0:self.x1]
        return l, r

    def disparity(self, l_bgr, r_bgr):
        gl = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gl.shape
        if DISP_SCALE != 1.0:
            sz = (int(w * DISP_SCALE), int(h * DISP_SCALE))
            gl = cv2.resize(gl, sz, interpolation=cv2.INTER_AREA)
            gr = cv2.resize(gr, sz, interpolation=cv2.INTER_AREA)
        d = self.sgbm.compute(gl, gr).astype(np.float32) / 16.0
        d[d <= 0.5] = np.nan
        if DISP_SCALE != 1.0:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST) / DISP_SCALE
        return d

    def point_3d(self, u, v, disp):
        z = self.f * self.baseline / disp * DEPTH_SCALE
        return np.array([(u - self.cx) * z / self.f,
                         (v - self.cy) * z / self.f,
                         z], dtype=np.float64)


def box_disparity(disp, box):
    x1, y1, x2, y2 = box
    h, w = disp.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    mx = int(bw * (1 - INNER_FRAC) / 2)
    my = int(bh * (1 - INNER_FRAC) / 2)
    roi = disp[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if roi.size == 0:
        roi = disp[y1:y2, x1:x2]
    v = roi[np.isfinite(roi)]
    if v.size < max(5, MIN_VALID_FRAC * roi.size):
        return None
    return float(np.median(v))


class GroundPlane:
    """Estimasi bidang tanah n.P = d0 di frame kamera (n menghadap ke atas) -> H & tilt otomatis."""

    def __init__(self):
        self.n = None
        self.d0 = None
        self.rng = np.random.default_rng(0)

    def update(self, disp, rig, exclude_boxes):
        h, w = disp.shape
        ys, xs = np.mgrid[int(h * GROUND_ROW_START):h:GROUND_STRIDE, 0:w:GROUND_STRIDE]
        d = disp[ys, xs]
        ok = np.isfinite(d) & (d > 1.0)
        for (bx1, by1, bx2, by2) in exclude_boxes:
            ok &= ~((xs >= bx1 - 5) & (xs <= bx2 + 5) & (ys >= by1 - 5) & (ys <= by2 + 5))
        d, xs, ys = d[ok], xs[ok].astype(np.float64), ys[ok].astype(np.float64)
        z = rig.f * rig.baseline / d * DEPTH_SCALE
        keep = z < MAX_RANGE_CM
        z, xs, ys = z[keep], xs[keep], ys[keep]
        if z.size < 200:
            return
        P = np.stack([(xs - rig.cx) * z / rig.f, (ys - rig.cy) * z / rig.f, z], axis=1)

        best_in, best_mask = 0, None
        for _ in range(RANSAC_ITERS):
            idx = self.rng.choice(len(P), 3, replace=False)
            p0, p1, p2 = P[idx]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            mask = np.abs((P - p0) @ n) < RANSAC_THRESH_CM
            c = int(mask.sum())
            if c > best_in:
                best_in, best_mask = c, mask
        if best_mask is None or best_in < GROUND_MIN_INLIER_FRAC * len(P):
            return

        Q = P[best_mask]
        c = Q.mean(axis=0)
        n = np.linalg.svd(Q - c)[2][-1]
        if n[1] > 0:                      # normal menghadap ke atas (Y kamera ke bawah)
            n = -n
        d0 = float(n @ c)

        if self.n is None:
            self.n, self.d0 = n, d0
        else:
            self.n = (1 - GROUND_EMA) * self.n + GROUND_EMA * n
            self.n /= np.linalg.norm(self.n)
            self.d0 = (1 - GROUND_EMA) * self.d0 + GROUND_EMA * d0

    @property
    def ready(self):
        return self.n is not None

    @property
    def height_cm(self):
        return -self.d0

    @property
    def tilt_deg(self):
        return float(np.degrees(np.arctan2(-self.n[2], -self.n[1])))

    def ground_distance(self, P):
        s = float(self.n @ P - self.d0)               # tinggi titik di atas bidang tanah
        Pg = P - s * self.n                            # proyeksi ke tanah
        foot = self.d0 * self.n                        # titik tanah tepat di bawah kamera
        return float(np.linalg.norm(Pg - foot)), s


def draw_label(img, text, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS_PATH, help="bobot YOLOv8n (.pt)")
    ap.add_argument("--calib", default=CALIB_PATH, help="hasil kalibrasi stereo (.npz)")
    ap.add_argument("--model", default="auto", choices=["auto", "pinhole", "fisheye"],
                    help="model lensa; auto = dari jumlah koefisien distorsi")
    ap.add_argument("--source", default=str(CAM_INDEX), help="indeks kamera atau path video stereo SBS")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="mis. 0 untuk GPU Jetson")
    ap.add_argument("--ground", action="store_true", help="estimasi H/tilt otomatis dari bidang tanah")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="hitung ulang rektifikasi, abaikan R1/P1 di .npz")
    ap.add_argument("--debug", action="store_true", help="tampilkan frame mentah + info diagnosis")
    ap.add_argument("--log", default=None, help="simpan hasil ke CSV")
    args = ap.parse_args()

    print(f"[INFO] {Path(__file__).name} | versi program: v5 (pinhole/fisheye otomatis)")
    rig = StereoRig(args.calib, args.model)
    if args.recompute and rig.saved is not None:
        rig.build_maps(skip_saved=True)
    model = YOLO(args.weights)
    ground = GroundPlane()
    use_ground = args.ground

    src = int(args.source) if args.source.isdigit() else args.source
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(src, backend) if isinstance(src, int) else cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit("Kamera/video tidak bisa dibuka")
    for _ in range(15):                    # buang frame awal (sering hitam)
        cap.read()
    checked = False

    writer, logf = None, None
    if args.log:
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "kelas", "conf", "disp_px", "X_cm", "Y_cm", "Z_cm", "D3D_cm",
                         "D_tanah_cm", "tinggi_obj_cm", "H_est_cm", "tilt_est_deg"])

    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            print(f"[PERINGATAN] ukuran frame {frame.shape[1]}x{frame.shape[0]} != "
                  f"{FRAME_W}x{FRAME_H}, di-resize (geometri kalibrasi bisa tidak cocok)") \
                if not checked else None
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        left, right = rig.rectify(frame)

        if not checked:
            checked = True
            raw_mean = float(frame.mean())
            rect_mean = float(left.mean())
            print(f"[DIAG] frame mentah {frame.shape[1]}x{frame.shape[0]}, rata-rata piksel = {raw_mean:.1f} | "
                  f"hasil rektifikasi (kiri) = {rect_mean:.1f}")
            if raw_mean < 3:
                print("[DIAG] Frame mentah sendiri hitam -> masalah kamera/driver (cek indeks kamera, "
                      "program lain yang memakai kamera, atau coba --source 1)")
            elif rect_mean < 3 and rig.saved is not None and rig.maps_saved_in_use():
                print("[DIAG] Frame mentah ada gambar tapi hasil rektifikasi hitam -> "
                      "beralih ke rektifikasi hitung ulang")
                rig.build_maps(skip_saved=True)
                left, right = rig.rectify(frame)
                print(f"[DIAG] setelah hitung ulang, rata-rata piksel = {float(left.mean()):.1f}")

        if args.debug and not args.no_show:
            cv2.imshow("RAW (debug)", cv2.resize(frame, (FRAME_W // 3, FRAME_H // 3)))

        disp = rig.disparity(left, right)

        res = model.predict(left, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int) if len(res.boxes) else []
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else []
        clss = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else []

        if use_ground:
            ground.update(disp, rig, [tuple(b) for b in boxes])

        vis = left.copy()
        now = time.time()
        for b, cf, ci in zip(boxes, confs, clss):
            x1, y1, x2, y2 = [int(v) for v in b]
            dpx = box_disparity(disp, (x1, y1, x2, y2))
            name = model.names[int(ci)]
            if dpx is None:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
                draw_label(vis, f"{name} {cf:.2f} | disp tak valid", (x1, max(15, y1 - 6)), (0, 165, 255))
                continue

            u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            P = rig.point_3d(u, v, dpx)
            D3 = float(np.linalg.norm(P))
            if P[2] > MAX_RANGE_CM:
                continue

            d_ground = s_obj = None
            line2 = ""
            if use_ground and ground.ready:
                d_ground, s_obj = ground.ground_distance(P)
                line2 = f"tanah {d_ground:.0f} cm"

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(vis, f"{name} {cf:.2f} | D={D3:.1f} cm  Z={P[2]:.1f}", (x1, max(15, y1 - 6)), (0, 255, 0))
            if line2:
                draw_label(vis, line2, (x1, y2 + 16), (0, 255, 255))

            if writer:
                writer.writerow([f"{now:.3f}", name, f"{cf:.3f}", f"{dpx:.2f}",
                                 f"{P[0]:.2f}", f"{P[1]:.2f}", f"{P[2]:.2f}", f"{D3:.2f}",
                                 "" if d_ground is None else f"{d_ground:.2f}",
                                 "" if s_obj is None else f"{s_obj:.2f}",
                                 f"{ground.height_cm:.2f}" if ground.ready else "",
                                 f"{ground.tilt_deg:.2f}" if ground.ready else ""])

        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now
        info = f"FPS {fps:.1f}"
        if use_ground and ground.ready:
            info += f" | H~{ground.height_cm:.1f} cm  tilt~{ground.tilt_deg:.1f} deg (otomatis)"
        draw_label(vis, info, (10, 20), (255, 255, 255))

        if not args.no_show:
            cv2.imshow("FOD detect + distance", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("g"):
                use_ground = not use_ground
            if k == ord("s"):
                cv2.imwrite(f"cap_{int(now)}.png", vis)

    cap.release()
    if logf:
        logf.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()