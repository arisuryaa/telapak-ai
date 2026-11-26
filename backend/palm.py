"""palm.py - menemukan tangan di foto dan memotong bagian tengah telapak (ROI).

Alurnya:
1. MediaPipe Hands mencari 21 titik tangan.
2. Foto diputar sampai garis pangkal telunjuk (titik 5) ke pangkal kelingking (titik 17) mendatar.
3. Kotak selebar telapak di bawah garis itu dipotong, diubah ke abu-abu, lalu diperkecil ke 128x128.
"""

import cv2
import mediapipe as mp
import numpy as np

UKURAN_ROI = 128  # ukuran ROI dalam piksel (128x128)
LEBAR_TELAPAK_MIN = 40  # piksel; kalau lebih kecil, tangan terlalu jauh dari kamera

# static_image_mode=True karena setiap permintaan berisi satu foto, bukan video.
_hands = mp.solutions.hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6
)


def deteksi_tangan(foto):
    """Cari tangan di foto (BGR).

    Kembalikan (titik, label): 21 titik tangan dan label "Left"/"Right" dari MediaPipe.
    Kembalikan None kalau tidak ada tangan.
    """
    hasil = _hands.process(cv2.cvtColor(foto, cv2.COLOR_BGR2RGB))
    if not hasil.multi_hand_landmarks:
        return None
    titik = hasil.multi_hand_landmarks[0].landmark
    label = hasil.multi_handedness[0].classification[0].label
    return titik, label


def ke_piksel(titik, nomor, lebar, tinggi):
    """Ubah titik MediaPipe (0..1) menjadi koordinat piksel [x, y]."""
    return np.array([titik[nomor].x * lebar, titik[nomor].y * tinggi])


def potong_roi(foto, titik):
    """Potong ROI telapak dari foto.

    Kembalikan dict berisi:
      roi        gambar abu-abu 128x128 (yang nanti masuk ke model)
      potongan   potongan berwarna sebelum diubah ke abu-abu
      foto_lurus foto yang sudah diputar
      kotak      (x1, y1, x2, y2) kotak ROI pada foto_lurus
      garis      dua ujung garis pangkal jari pada foto_lurus
      sudut      berapa derajat foto diputar
    Kembalikan None kalau tangan terlalu kecil atau kotaknya keluar dari foto.
    """
    tinggi, lebar = foto.shape[:2]
    p5 = ke_piksel(titik, 5, lebar, tinggi)  # pangkal telunjuk
    p17 = ke_piksel(titik, 17, lebar, tinggi)  # pangkal kelingking
    p0 = ke_piksel(titik, 0, lebar, tinggi)  # pergelangan

    lebar_telapak = float(np.linalg.norm(p17 - p5))
    if lebar_telapak < LEBAR_TELAPAK_MIN:
        return None

    # Putar foto supaya garis p5-p17 mendatar.
    sudut = float(np.degrees(np.arctan2(p17[1] - p5[1], p17[0] - p5[0])))
    pusat = (p5 + p17) / 2
    putar = cv2.getRotationMatrix2D((float(pusat[0]), float(pusat[1])), sudut, 1.0)
    foto_lurus = cv2.warpAffine(foto, putar, (lebar, tinggi))

    # Setelah diputar, apakah pergelangan ada di bawah atau di atas garis?
    p0_lurus = putar @ np.array([p0[0], p0[1], 1.0])
    arah = 1 if p0_lurus[1] > pusat[1] else -1

    # Kotak persegi selebar telapak, dari 0,15 sampai 1,15 lebar telapak di bawah garis.
    s = lebar_telapak
    y_a = pusat[1] + arah * 0.15 * s
    y_b = pusat[1] + arah * 1.15 * s
    x1, x2 = int(pusat[0] - s / 2), int(pusat[0] + s / 2)
    y1, y2 = int(min(y_a, y_b)), int(max(y_a, y_b))
    if x1 < 0 or y1 < 0 or x2 > lebar or y2 > tinggi:
        return None

    potongan = foto_lurus[y1:y2, x1:x2]
    if arah == -1:  # tangan terbalik: putar 180 derajat supaya jari selalu di atas
        potongan = cv2.rotate(potongan, cv2.ROTATE_180)
    abu = cv2.cvtColor(potongan, cv2.COLOR_BGR2GRAY)

    garis = [putar @ np.array([p[0], p[1], 1.0]) for p in (p5, p17)]
    return {
        "roi": cv2.resize(abu, (UKURAN_ROI, UKURAN_ROI)),
        "potongan": potongan.copy(),
        "foto_lurus": foto_lurus,
        "kotak": (x1, y1, x2, y2),
        "garis": [(int(x), int(y)) for x, y in garis],
        "sudut": sudut,
    }


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def pertegas_kontras(roi):
    """Pertegas garis telapak dengan CLAHE (ekualisasi histogram per kotak kecil)."""
    return _CLAHE.apply(roi)
