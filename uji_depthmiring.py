import cv2
import numpy as np
import csv

# ============================================================
# KONFIGURASI
# ============================================================

CALIB_FILE = "stereo_calib_fisheye2.npz"
CAMERA_INDEX = 0

FRAME_WIDTH = 2560        # gabungan kiri+kanan
FRAME_HEIGHT = 720

OUTPUT_CSV = "hasil_uji_depth_miring.csv"

FIX_RIGHT_DISTORTION = False
FIX_PROJECTION_MATRIX = True

# --- Geometri pemasangan kamera ---
CAM_HEIGHT_M = 1.0        # H : tinggi pusat lensa KIRI dari tanah (ukur!)
TILT_DEG = 25.0           # theta : sudut menunduk dari horizontal

# --- Akurasi stereo ---
RECT_F_SCALE = 1.5
MIN_DISTANCE_M = 0.35     # jarak optik (Z) terdekat -> NUM_DISP otomatis
MAX_DEPTH_M = 10.0

PREVIEW_SCALE = 0.5
MEASURE_SCALE = 1.0
MEASURE_FRAMES = 10 

BLOCK_SIZE = 7
SMOOTH_CN = 2             # diturunkan dari 3: bidang tanah miring cenderung berundak jika terlalu besar
PREVIEW_HH = True
CLEAN_DISPLAY = True
MAX_HOLE_PX = 1500
DISPLAY_SCALE = 0.6

# ============================================================
# LOAD KALIBRASI
# ============================================================

data = np.load(CALIB_FILE)

mtx_left = data["mtx_left"].astype(np.float64)
dist_left = data["dist_left"].astype(np.float64)
mtx_right = data["mtx_right"].astype(np.float64)
dist_right = data["dist_right"].astype(np.float64)

img_shape = tuple(int(x) for x in data["img_shape"])   # (width, height) PER MATA
baseline = float(data["baseline"])

print("=" * 60)
print("        UJI DEPTH FISHEYE - KAMERA MIRING")
print("=" * 60)
print(f"\nBaseline kalibrasi : {baseline * 100:.3f} cm")
print(f"Resolusi kalibrasi : {img_shape[0]}x{img_shape[1]} (per mata)")

if FIX_RIGHT_DISTORTION:
    dist_right[2, 0] = 0.0
    dist_right[3, 0] = 0.0

R1 = data["R1"].astype(np.float64)
R2 = data["R2"].astype(np.float64)

# ============================================================
# P1 / P2
# ============================================================

if FIX_PROJECTION_MATRIX:
    common_f = (mtx_left[0, 0] + mtx_left[1, 1] +
                mtx_right[0, 0] + mtx_right[1, 1]) / 4.0
    cx0 = (mtx_left[0, 2] + mtx_right[0, 2]) / 2.0
    cy0 = (mtx_left[1, 2] + mtx_right[1, 2]) / 2.0

    P1 = np.array([[common_f, 0, cx0, 0],
                   [0, common_f, cy0, 0],
                   [0, 0, 1, 0]], dtype=np.float64)
    P2 = np.array([[common_f, 0, cx0, -common_f * baseline],
                   [0, common_f, cy0, 0],
                   [0, 0, 1, 0]], dtype=np.float64)
else:
    P1 = data["P1"].astype(np.float64)
    P2 = data["P2"].astype(np.float64)

for P in (P1, P2):
    P[0, 0] *= RECT_F_SCALE
    P[1, 1] *= RECT_F_SCALE
P2[0, 3] *= RECT_F_SCALE

FOCAL = float(P1[0, 0])
BASE_EFF = float(-P2[0, 3] / P2[0, 0])
FB = FOCAL * BASE_EFF
CX, CY = float(P1[0, 2]), float(P1[1, 2])

NUM_DISP_FULL = int(np.ceil(FB / MIN_DISTANCE_M / 16.0)) * 16
Z_MIN = FB / NUM_DISP_FULL

print(f"\nfocal rectified = {FOCAL:.2f} px (x{RECT_F_SCALE}), baseline = {BASE_EFF*100:.3f} cm")
print(f"NUM_DISP (resolusi penuh) = {NUM_DISP_FULL} -> Zmin = {Z_MIN*100:.1f} cm")

# ============================================================
# GEOMETRI KAMERA MIRING
# ============================================================
# Frame kamera (rectified): x kanan, y bawah, z sepanjang sumbu optik.
# Dunia: D = jarak horizontal ke depan (dari titik tanah tepat di bawah lensa kiri),
#        X = lateral (kanan +), h = tinggi di atas tanah.
# Pitch menunduk theta, roll = 0.

H = CAM_HEIGHT_M
TH = np.radians(TILT_DEG)
S, C = np.sin(TH), np.cos(TH)


def to_world(x, y, z):
    """(piksel x,y ; depth stereo Z) -> X, D, h, R."""
    a = (x - CX) / FOCAL
    b = (y - CY) / FOCAL
    X = a * z
    D = z * (C - b * S)
    h = H - z * (S + b * C)
    R = z * np.sqrt(1.0 + a * a + b * b)
    return X, D, h, R


def expected_ground_depth(y):
    """Z yang diharapkan jika piksel baris y jatuh di TANAH (h=0)."""
    b = (y - CY) / FOCAL
    den = S + b * C
    return H / den if den > 1e-6 else np.nan


def ground_distance_from_row(y):
    """D (m) hanya dari baris piksel, tanpa stereo, asumsi titik di tanah."""
    b = (y - CY) / FOCAL
    den = S + b * C
    return H * (C - b * S) / den if den > 1e-6 else np.nan


def expected_Z_from_truth(D_true, h_true):
    """Z kamera yang seharusnya untuk titik pada (D, h) hasil ukur manual."""
    return D_true * C + (H - h_true) * S


def horizon_row():
    return CY - FOCAL * np.tan(TH)


print("\n--- Geometri ---")
print(f"H = {H:.2f} m | theta = {TILT_DEG:.1f} deg | horizon pada y = {horizon_row():.0f} px")
print("Jarak tanah (D) per baris gambar (referensi merencanakan titik uji):")
for yy in np.linspace(img_shape[1] * 0.35, img_shape[1] - 1, 6):
    zg = expected_ground_depth(yy)
    if np.isfinite(zg):
        print(f"  y={yy:5.0f}px -> Zground={zg:5.2f} m | D={ground_distance_from_row(yy):5.2f} m "
              f"| d={FB / zg:5.1f}px")
print("Error 0.5 px pada disparity -> dZ = Z^2/(f*B)*0.5")
for z in (1.0, 2.0, 3.0, 4.0):
    print(f"  Z={z:.1f} m -> dZ = {z**2 / FB * 0.5 * 100:5.1f} cm")

# ============================================================
# PETA RECTIFICATION
# ============================================================

map_left_x, map_left_y = cv2.fisheye.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1, img_shape, cv2.CV_32FC1)
map_right_x, map_right_y = cv2.fisheye.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2, img_shape, cv2.CV_32FC1)

_ones = np.full((img_shape[1], img_shape[0]), 255, np.uint8)
VALID_MASK = cv2.remap(_ones, map_left_x, map_left_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
VALID_MASK = cv2.erode(VALID_MASK, np.ones((21, 21), np.uint8)) > 0

# ============================================================
# STEREO MATCHER
# ============================================================

def build_matcher(scale, hh=False):
    nd = max(16, int(np.ceil(NUM_DISP_FULL * scale / 16.0)) * 16)
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=BLOCK_SIZE,
        P1=8 * SMOOTH_CN * BLOCK_SIZE ** 2,
        P2=32 * SMOOTH_CN * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH if hh else cv2.STEREO_SGBM_MODE_SGBM_3WAY)


matcher_preview = build_matcher(PREVIEW_SCALE, hh=PREVIEW_HH)
matcher_measure = build_matcher(MEASURE_SCALE, hh=False)


def compute_disparity(gray_l, gray_r, matcher, s):
    if s != 1.0:
        gl = cv2.resize(gray_l, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gr = cv2.resize(gray_r, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        gl, gr = gray_l, gray_r
    gl = cv2.GaussianBlur(gl, (3, 3), 0)
    gr = cv2.GaussianBlur(gr, (3, 3), 0)

    disp = matcher.compute(gl, gr).astype(np.float32) / 16.0
    disp[disp <= 0.5] = 0
    disp = cv2.medianBlur(disp, 5)

    if s != 1.0:
        disp = cv2.resize(disp, (gray_l.shape[1], gray_l.shape[0]),
                          interpolation=cv2.INTER_NEAREST) / s
    disp[disp <= 0.5 / s] = np.nan
    if disp.shape == VALID_MASK.shape:
        disp[~VALID_MASK] = np.nan
    return disp


def clean_for_display(disp):
    valid = np.isfinite(disp)
    d = np.where(valid, disp, 0).astype(np.float32)

    region = np.ones(disp.shape, bool)
    if disp.shape == VALID_MASK.shape:
        region &= VALID_MASK
    region[:, :NUM_DISP_FULL] = False

    hole = ((~valid) & region).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    small_ids = np.where(stats[1:, cv2.CC_STAT_AREA] <= MAX_HOLE_PX)[0] + 1
    fill = np.isin(lab, small_ids)

    d8 = np.clip(d / NUM_DISP_FULL * 255.0, 0, 255).astype(np.uint8)
    if fill.any():
        d8 = cv2.inpaint(d8, fill.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    d8 = cv2.bilateralFilter(d8, 9, 40, 9)

    out = d8.astype(np.float32) * NUM_DISP_FULL / 255.0
    out[~(valid | fill)] = np.nan
    return out


def get_depth(disp_full, x, y, win=7):
    """Median disparity di jendela kecil -> (Z m, disparity px)."""
    h_, w_ = disp_full.shape
    x0, x1 = max(0, x - win), min(w_, x + win + 1)
    y0, y1 = max(0, y - win), min(h_, y + win + 1)
    patch = disp_full[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    if vals.size < 0.3 * patch.size:
        return None, None
    d = float(np.median(vals))
    z = FB / d
    if not np.isfinite(z) or z <= 0 or z > MAX_DEPTH_M:
        return None, None
    return z, d


def rectify_gray(frame):
    half = frame.shape[1] // 2
    left_rect = cv2.remap(frame[:, :half], map_left_x, map_left_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(frame[:, half:], map_right_x, map_right_y, cv2.INTER_LINEAR)
    return (left_rect,
            cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY))


# ============================================================
# BUKA KAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("\nERROR: Kamera gagal dibuka.")
    exit()

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nResolusi kamera aktif : {actual_w}x{actual_h}")

if (actual_w // 2, actual_h) != img_shape:
    print("\nPERINGATAN: Resolusi kamera berbeda dari saat kalibrasi.")

# ============================================================
# MOUSE
# ============================================================

WIN_LEFT = "Rectified LEFT (klik untuk cek depth)"
WIN_DISP = "Disparity Map"
clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE))


cv2.namedWindow(WIN_LEFT)
cv2.setMouseCallback(WIN_LEFT, mouse_callback)

# ============================================================
# PENGUKURAN BURST
# ============================================================

def measure_burst(cx, cy):
    zs = []
    for _ in range(MEASURE_FRAMES):
        ret, fr = cap.read()
        if not ret:
            continue
        _, gl, gr = rectify_gray(fr)
        disp = compute_disparity(gl, gr, matcher_measure, MEASURE_SCALE)
        z, _ = get_depth(disp, cx, cy)
        if z is not None:
            zs.append(z)
    if len(zs) < max(3, MEASURE_FRAMES // 2):
        return None, None, len(zs)
    return float(np.median(zs)), float(np.std(zs)), len(zs)


def stats_line(name, err, sgn, pct=None):
    s = (f"{name:<10}: MAE={np.mean(err):7.2f} cm | RMSE={np.sqrt(np.mean(err**2)):7.2f} cm | "
         f"Bias={np.mean(sgn):+7.2f} cm")
    if pct is not None:
        s += f" | MAPE={np.mean(pct):5.2f}%"
    return s


def print_summary(ms, final=False):
    print("\n" + "=" * 78)
    print("HASIL AKHIR UJI DEPTH (KAMERA MIRING)" if final else "RINGKASAN SEMENTARA")
    print("=" * 78)
    print(f"H = {H:.2f} m | theta = {TILT_DEG:.1f} deg")
    for i, m in enumerate(ms):
        print(f"{i+1:2d}. D_asli={m['D_true_cm']:6.1f} | D_stereo={m['D_stereo_cm']:6.1f} "
              f"(err {m['err_D_cm']:+6.1f}) | h_asli={m['h_true_cm']:5.1f} | "
              f"h_stereo={m['h_stereo_cm']:6.1f} | Z_stereo={m['Z_stereo_cm']:6.1f} "
              f"(Z_ideal {m['Z_true_cm']:6.1f}) | std={m['Z_std_cm']:.1f}")

    eD = np.array([m["D_stereo_cm"] - m["D_true_cm"] for m in ms])
    eH = np.array([m["h_stereo_cm"] - m["h_true_cm"] for m in ms])
    eZ = np.array([m["Z_stereo_cm"] - m["Z_true_cm"] for m in ms])
    pD = np.array([abs(m["err_D_cm"]) / m["D_true_cm"] * 100 if m["D_true_cm"] else 0 for m in ms])
    pZ = np.array([abs(e) / m["Z_true_cm"] * 100 for e, m in zip(eZ, ms)])

    print(f"\nJumlah data : {len(ms)}")
    print(stats_line("D (horiz)", np.abs(eD), eD, pD))
    print(stats_line("h (tinggi)", np.abs(eH), eH))
    print(stats_line("Z (optik)", np.abs(eZ), eZ, pZ))

    gp = np.array([m["D_ground_cm"] - m["D_true_cm"] for m in ms
                   if np.isfinite(m["D_ground_cm"]) and abs(m["h_true_cm"]) < 1e-6])
    if gp.size:
        print(stats_line("D ground", np.abs(gp), gp) + "   (tanpa stereo, hanya baris piksel)")

    if len(ms) >= 3:
        a = np.array([m["D_true_cm"] for m in ms])
        t = np.array([m["D_stereo_cm"] for m in ms])
        slope, intercept = np.polyfit(a, t, 1)
        print(f"Fit D_stereo = {slope:.4f} * D_asli + {intercept:.2f}")

    print("\nPer rentang jarak horizontal (D asli):")
    bins = [(0, 150), (150, 250), (250, 400), (400, 1e9)]
    for lo, hi in bins:
        idx = [i for i, m in enumerate(ms) if lo <= m["D_true_cm"] < hi]
        if idx:
            e = np.abs(eD[idx])
            lab = f"{lo:.0f}-{hi:.0f} cm" if hi < 1e8 else f">{lo:.0f} cm"
            print(f"  {lab:<12} n={len(idx):2d} | MAE={e.mean():6.2f} cm | RMSE={np.sqrt(np.mean(e**2)):6.2f} cm")
    print("=" * 78)


measurements = []

print("\n" + "=" * 60)
print("CARA PAKAI:")
print("=" * 60)
print("1. Ukur D_asli: jarak HORIZONTAL (di lantai) dari titik tepat di")
print("   bawah LENSA KIRI (pakai unting-unting) ke titik dasar objek.")
print("2. Taruh objek BERTEKSTUR. Klik titik yang ingin diuji.")
print("3. Tekan 's' -> input D_asli (cm) dan h_asli (cm).")
print("   h_asli = tinggi titik yang diklik di atas lantai (0 jika di lantai).")
print("4. Ulangi pada beberapa jarak (mis. 100, 150, 200, 250, 300 cm).")
print("5. 'r' ringkasan, ESC keluar & simpan CSV.")
print("=" * 60 + "\n")

# ============================================================
# LOOP UTAMA
# ============================================================

while True:

    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    left_rect, gray_left, gray_right = rectify_gray(frame)
    disparity = compute_disparity(gray_left, gray_right, matcher_preview, PREVIEW_SCALE)

    disp_vis = clean_for_display(disparity) if CLEAN_DISPLAY else disparity
    vis = np.nan_to_num(disp_vis, nan=0.0) / NUM_DISP_FULL * 255.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    disp_display = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    disp_display[~np.isfinite(disp_vis)] = (0, 0, 0)
    cv2.putText(disp_display, "merah=dekat | biru=jauh | hitam=invalid",
                (10, disp_display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    left_display = left_rect.copy()

    # garis horizon teoritis
    hy = int(round(horizon_row()))
    if 0 <= hy < left_display.shape[0]:
        cv2.line(left_display, (0, hy), (left_display.shape[1], hy), (255, 128, 0), 1)
        cv2.putText(left_display, "horizon", (10, hy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

    if clicked_point is not None:
        cx, cy = clicked_point
        if 0 <= cy < disparity.shape[0] and 0 <= cx < disparity.shape[1]:
            depth_val, disp_val = get_depth(disparity, cx, cy)

            cv2.circle(left_display, (cx, cy), 8, (0, 0, 255), 2)
            cv2.rectangle(left_display, (cx - 7, cy - 7), (cx + 7, cy + 7), (0, 255, 255), 1)
            cv2.circle(disp_display, (cx, cy), 8, (255, 255, 255), 2)

            if depth_val is not None:
                X, D, h, R = to_world(cx, cy, depth_val)
                zg = expected_ground_depth(cy)
                dg = ground_distance_from_row(cy)
                line1 = (f"Z={depth_val:.2f} m | D={D:.2f} m | X={X:+.2f} m | "
                         f"h={h*100:.0f} cm | d={disp_val:.1f}px")
                line2 = f"Jika di tanah: Zg={zg:.2f} m | Dg={dg:.2f} m"
                color = (0, 255, 0)
            else:
                line1 = "Depth: tidak valid (area polos / terlalu dekat / terlalu jauh)"
                line2 = ""
                color = (0, 0, 255)

            for i, t in enumerate((line1, line2)):
                if not t:
                    continue
                yy = 30 + i * 30
                cv2.putText(left_display, t, (10, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(left_display, t, (10, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(left_display,
                "Klik objek | 's' simpan | 'r' ringkasan | ESC keluar",
                (10, left_display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WIN_LEFT, cv2.resize(left_display, None,
                                    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
    cv2.imshow(WIN_DISP, cv2.resize(disp_display, None,
                                    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord('s'):
        if clicked_point is None:
            print("\n[GAGAL] Klik dulu titik pada objek sebelum menyimpan.")
            continue

        cx, cy = clicked_point
        print(f"\nMengukur {MEASURE_FRAMES} frame pada resolusi penuh... jangan bergerak.")
        z_med, z_std, n_ok = measure_burst(cx, cy)

        if z_med is None:
            print(f"[GAGAL] Hanya {n_ok} frame valid. Klik area yang lebih bertekstur.")
            continue

        X, D, h, R = to_world(cx, cy, z_med)
        print(f"Terukur: Z={z_med*100:.1f} cm | D={D*100:.1f} cm | h={h*100:.1f} cm "
              f"(std Z {z_std*100:.1f} cm, {n_ok} frame)")

        try:
            D_true = float(input("Jarak HORIZONTAL sebenarnya D (cm): "))
            h_in = input("Tinggi titik di atas lantai h (cm) [Enter = 0]: ").strip()
            h_true = float(h_in) if h_in else 0.0
        except ValueError:
            print("[GAGAL] Input harus berupa angka.")
            continue

        Z_true = expected_Z_from_truth(D_true / 100.0, h_true / 100.0) * 100.0
        dg = ground_distance_from_row(cy)
        dg_cm = dg * 100 if np.isfinite(dg) else float("nan")

        rec = {
            "D_true_cm": D_true,
            "D_stereo_cm": D * 100,
            "err_D_cm": D * 100 - D_true,
            "h_true_cm": h_true,
            "h_stereo_cm": h * 100,
            "err_h_cm": h * 100 - h_true,
            "Z_true_cm": Z_true,
            "Z_stereo_cm": z_med * 100,
            "err_Z_cm": z_med * 100 - Z_true,
            "D_ground_cm": dg_cm,
            "Z_std_cm": z_std * 100,
            "X_stereo_cm": X * 100,
            "pix_x": cx,
            "pix_y": cy,
        }
        measurements.append(rec)

        print(f"[TERSIMPAN] D: asli={D_true:.1f} stereo={D*100:.1f} (err {rec['err_D_cm']:+.1f}) | "
              f"Z: ideal={Z_true:.1f} stereo={z_med*100:.1f} (err {rec['err_Z_cm']:+.1f}) | "
              f"Total data: {len(measurements)}")

    elif key == ord('r'):
        if not measurements:
            print("\nBelum ada data pengukuran tersimpan.")
            continue
        print_summary(measurements)

# ============================================================
# SELESAI
# ============================================================

cap.release()
cv2.destroyAllWindows()

if measurements:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measurements[0].keys()))
        writer.writeheader()
        writer.writerows(measurements)
    print_summary(measurements, final=True)
    print(f"Data disimpan ke : {OUTPUT_CSV}")
else:
    print("\nTidak ada data pengukuran yang disimpan.")