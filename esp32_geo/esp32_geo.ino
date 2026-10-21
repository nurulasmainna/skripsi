/*
  esp32_geo.ino
  ESP32 + GPS NEO-8M + MPU9250  ->  serial USB ke Jetson

  Keluaran (115200 baud, 10 Hz), satu baris per pesan:
      GEO,lat,lon,heading_deg,fix,sats,hdop
  contoh:
      GEO,-7.983900,112.621400,87.5,1,9,1.1
  Baris yang diawali '#' adalah pesan info (diabaikan program Jetson).

  Wiring:
    NEO-8M : TX -> GPIO16 (RX2) | RX -> GPIO17 (TX2) | VCC | GND   (9600 baud)
    MPU9250: SDA -> GPIO21 | SCL -> GPIO22 | VCC 3.3V | GND | AD0 ke GND (alamat 0x68)

  Library yang dibutuhkan: TinyGPSPlus (Mikal Hart). Selain itu tidak ada (MPU9250 dibaca langsung lewat I2C).
  Board: "ESP32 Dev Module".

  Heading = heading MAGNETIK (0 = utara magnet, searah jarum jam), dengan kompensasi kemiringan
  memakai akselerometer. Koreksi deklinasi dan offset pasang dilakukan di Jetson
  (--declination, --yaw-offset), atau lewat HEADING_OFFSET_DEG di bawah.

  Kalibrasi magnetometer (WAJIB): kirim karakter 'c' lewat Serial Monitor (115200), lalu putar
  sensor ke segala arah (gerakan angka 8) selama 30 detik. Hasil disimpan di flash (tetap
  setelah restart). Lakukan di tempat jauh dari besi/motor dan pasang di posisi akhir
  (di platform) bila memungkinkan.
*/

#include <Wire.h>
#include <Preferences.h>
#include <TinyGPSPlus.h>

// ---------------- konfigurasi ----------------
#define SERIAL_BAUD        115200
#define GPS_RX_PIN         16
#define GPS_TX_PIN         17
#define GPS_BAUD           9600
#define I2C_SDA            21
#define I2C_SCL            22

// Sumbu sensor MPU9250 yang menghadap MAJU (arah pandang kamera). Default: sumbu +X.
#define FWD_X              1.0f
#define FWD_Y              0.0f
#define FWD_Z              0.0f

#define HEADING_SIGN       1.0f     // ganti -1.0f jika heading malah turun saat diputar searah jarum jam
#define HEADING_OFFSET_DEG 0.0f     // koreksi tetap (derajat)

#define OUT_PERIOD_MS      100
#define IMU_PERIOD_MS      10
#define CAL_DURATION_MS    30000
#define FILTER_ALPHA       0.15f
// ---------------------------------------------

TinyGPSPlus gps;
Preferences prefs;

uint8_t mpuAddr = 0x68;
const uint8_t AK_ADDR = 0x0C;
float asa[3] = {1, 1, 1};

bool magCalibrated = false;
float magOff[3] = {0, 0, 0};
float magScl[3] = {1, 1, 1};

float aF[3] = {0, 0, 1}, mF[3] = {1, 0, 0};
bool filtInit = false;
float sinF = 0, cosF = 1;
float headingDeg = 0;

// ---------------- I2C helper ----------------
void writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readRegs(uint8_t addr, uint8_t reg, uint8_t n, uint8_t *buf) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

// ---------------- MPU9250 ----------------
bool initMPU() {
  const uint8_t addrs[2] = {0x68, 0x69};
  bool found = false;
  uint8_t id = 0;
  for (int i = 0; i < 2; i++) {
    if (readRegs(addrs[i], 0x75, 1, &id)) {
      if (id == 0x71 || id == 0x73) { mpuAddr = addrs[i]; found = true; break; }
      Serial.printf("# WHO_AM_I di 0x%02X = 0x%02X\n", addrs[i], id);
      if (id == 0x70) Serial.println("# Chip ini MPU6500 (TANPA magnetometer). Modul 'MPU9250' palsu/salah.");
    }
  }
  if (!found) return false;

  writeReg(mpuAddr, 0x6B, 0x01);   // bangun, clock PLL
  delay(100);
  writeReg(mpuAddr, 0x1C, 0x00);   // akselerometer +-2g
  writeReg(mpuAddr, 0x1D, 0x03);   // DLPF akselerometer
  writeReg(mpuAddr, 0x6A, 0x00);   // I2C master internal mati
  writeReg(mpuAddr, 0x37, 0x02);   // bypass: magnetometer langsung di bus I2C
  delay(10);

  uint8_t wia = 0;
  if (!readRegs(AK_ADDR, 0x00, 1, &wia) || wia != 0x48) {
    Serial.printf("# AK8963 tidak terdeteksi (WIA=0x%02X)\n", wia);
    return false;
  }
  uint8_t b[3];
  writeReg(AK_ADDR, 0x0A, 0x00); delay(10);
  writeReg(AK_ADDR, 0x0A, 0x0F); delay(10);          // fuse ROM
  readRegs(AK_ADDR, 0x10, 3, b);
  for (int i = 0; i < 3; i++) asa[i] = ((float)b[i] - 128.0f) / 256.0f + 1.0f;
  writeReg(AK_ADDR, 0x0A, 0x00); delay(10);
  writeReg(AK_ADDR, 0x0A, 0x16); delay(10);          // 16-bit, kontinu 100 Hz
  return true;
}

bool readAccel(float a[3]) {
  uint8_t b[6];
  if (!readRegs(mpuAddr, 0x3B, 6, b)) return false;
  a[0] = (int16_t)((b[0] << 8) | b[1]) / 16384.0f;
  a[1] = (int16_t)((b[2] << 8) | b[3]) / 16384.0f;
  a[2] = (int16_t)((b[4] << 8) | b[5]) / 16384.0f;
  return true;
}

// Magnetometer dipetakan ke sumbu akselerometer: (x,y,z)_acc = (my, mx, -mz)_mag. Satuan uT.
bool readMag(float m[3]) {
  uint8_t st1 = 0;
  if (!readRegs(AK_ADDR, 0x02, 1, &st1) || !(st1 & 0x01)) return false;
  uint8_t b[7];
  if (!readRegs(AK_ADDR, 0x03, 7, b)) return false;
  if (b[6] & 0x08) return false;                      // overflow
  float x = (int16_t)((b[1] << 8) | b[0]) * asa[0] * 0.15f;
  float y = (int16_t)((b[3] << 8) | b[2]) * asa[1] * 0.15f;
  float z = (int16_t)((b[5] << 8) | b[4]) * asa[2] * 0.15f;
  m[0] = y;
  m[1] = x;
  m[2] = -z;
  return true;
}

// ---------------- kalibrasi magnetometer ----------------
void computeCal(const float mn[3], const float mx[3]) {
  float rng[3], avg = 0;
  for (int i = 0; i < 3; i++) {
    magOff[i] = (mx[i] + mn[i]) / 2.0f;
    rng[i] = (mx[i] - mn[i]) / 2.0f;
    avg += rng[i] / 3.0f;
  }
  for (int i = 0; i < 3; i++) magScl[i] = avg / rng[i];
  magCalibrated = true;
}

void loadCal() {
  prefs.begin("magcal", false);
  if (prefs.getBool("ok", false)) {
    float mn[3], mx[3];
    for (int i = 0; i < 3; i++) {
      mn[i] = prefs.getFloat((String("mn") + i).c_str(), 0);
      mx[i] = prefs.getFloat((String("mx") + i).c_str(), 0);
    }
    computeCal(mn, mx);
    Serial.println("# Kalibrasi magnetometer dimuat dari flash");
  } else {
    Serial.println("# PERINGATAN: magnetometer BELUM dikalibrasi. Kirim 'c' lalu putar sensor 30 detik.");
  }
}

void calibrateMag() {
  Serial.println("# KALIBRASI: putar sensor ke segala arah (angka 8) selama 30 detik...");
  float mn[3] = {1e9f, 1e9f, 1e9f}, mx[3] = {-1e9f, -1e9f, -1e9f};
  uint32_t t0 = millis(), lastP = 0;
  while (millis() - t0 < CAL_DURATION_MS) {
    float m[3];
    if (readMag(m)) {
      for (int i = 0; i < 3; i++) {
        if (m[i] < mn[i]) mn[i] = m[i];
        if (m[i] > mx[i]) mx[i] = m[i];
      }
    }
    if (millis() - lastP > 5000) {
      lastP = millis();
      Serial.printf("# ... %lu detik tersisa\n", (CAL_DURATION_MS - (millis() - t0)) / 1000);
    }
    delay(10);
  }
  for (int i = 0; i < 3; i++) {
    if (mx[i] - mn[i] < 10.0f) {
      Serial.println("# KALIBRASI GAGAL: rentang sumbu terlalu kecil (putar lebih lengkap). Ulangi dengan 'c'.");
      return;
    }
  }
  computeCal(mn, mx);
  prefs.putBool("ok", true);
  for (int i = 0; i < 3; i++) {
    prefs.putFloat((String("mn") + i).c_str(), mn[i]);
    prefs.putFloat((String("mx") + i).c_str(), mx[i]);
  }
  Serial.printf("# Kalibrasi selesai. offset = %.1f %.1f %.1f uT | skala = %.3f %.3f %.3f\n",
                magOff[0], magOff[1], magOff[2], magScl[0], magScl[1], magScl[2]);
}

// ---------------- heading ----------------
void updateHeading() {
  float a[3];
  if (!readAccel(a)) return;
  float m[3];
  bool newMag = readMag(m);
  if (newMag && magCalibrated) {
    for (int i = 0; i < 3; i++) m[i] = (m[i] - magOff[i]) * magScl[i];
  }

  if (!filtInit) {
    for (int i = 0; i < 3; i++) aF[i] = a[i];
    if (newMag) { for (int i = 0; i < 3; i++) mF[i] = m[i]; filtInit = true; }
    else return;
  } else {
    for (int i = 0; i < 3; i++) aF[i] += FILTER_ALPHA * (a[i] - aF[i]);
    if (newMag) for (int i = 0; i < 3; i++) mF[i] += FILTER_ALPHA * (m[i] - mF[i]);
  }

  // up = arah akselerasi (specific force menunjuk ke atas saat diam)
  float an = sqrtf(aF[0] * aF[0] + aF[1] * aF[1] + aF[2] * aF[2]);
  if (an < 0.3f) return;
  float ux = aF[0] / an, uy = aF[1] / an, uz = aF[2] / an;

  // east = m x up ; north = up x east   (ENU, tangan kanan)
  float ex = mF[1] * uz - mF[2] * uy;
  float ey = mF[2] * ux - mF[0] * uz;
  float ez = mF[0] * uy - mF[1] * ux;
  float en = sqrtf(ex * ex + ey * ey + ez * ez);
  if (en < 1e-6f) return;
  ex /= en; ey /= en; ez /= en;
  float nx = uy * ez - uz * ey;
  float ny = uz * ex - ux * ez;
  float nz = ux * ey - uy * ex;

  float fe = FWD_X * ex + FWD_Y * ey + FWD_Z * ez;   // komponen "maju" ke timur
  float fn = FWD_X * nx + FWD_Y * ny + FWD_Z * nz;   // komponen "maju" ke utara
  float h = atan2f(fe, fn) * 180.0f / PI;            // searah jarum jam dari utara
  h = HEADING_SIGN * h + HEADING_OFFSET_DEG;

  float r = h * PI / 180.0f;                         // haluskan lewat sin/cos (aman di batas 0/360)
  sinF += FILTER_ALPHA * (sinf(r) - sinF);
  cosF += FILTER_ALPHA * (cosf(r) - cosF);
  float hh = atan2f(sinF, cosF) * 180.0f / PI;
  if (hh < 0) hh += 360.0f;
  headingDeg = hh;
}

// ---------------- output ----------------
void sendGeo() {
  bool fix = gps.location.isValid() && gps.location.age() < 2000;
  double lat = gps.location.isValid() ? gps.location.lat() : 0.0;
  double lon = gps.location.isValid() ? gps.location.lng() : 0.0;
  int sats = gps.satellites.isValid() ? (int)gps.satellites.value() : 0;
  double hdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.0;
  Serial.printf("GEO,%.6f,%.6f,%.1f,%d,%d,%.1f\n", lat, lon, headingDeg, fix ? 1 : 0, sats, hdop);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial2.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  delay(300);

  if (!initMPU()) {
    while (true) {
      Serial.println("# ERROR: MPU9250/magnetometer tidak terdeteksi. Cek wiring SDA=21 SCL=22 dan 3.3V.");
      delay(2000);
    }
  }
  loadCal();
  Serial.println("# Siap. Kirim 'c' untuk kalibrasi magnetometer.");
}

void loop() {
  while (Serial2.available()) gps.encode(Serial2.read());

  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' || c == 'C') calibrateMag();
  }

  static uint32_t tImu = 0, tOut = 0;
  uint32_t now = millis();
  if (now - tImu >= IMU_PERIOD_MS) { tImu = now; updateHeading(); }
  if (now - tOut >= OUT_PERIOD_MS) { tOut = now; sendGeo(); }
}
