"""validation.py - aturan tangan yang dicek di server sebelum foto dipakai.

Aturan yang sama juga dicek di browser (frontend/src/lib/validation.js).
"""

import cv2
import numpy as np

from palm import ke_piksel, pertegas_kontras

RASIO_JARI = 1.1  # ujung jari harus lebih jauh dari pergelangan daripada ruas tengahnya
RASIO_JEMPOL = 0.7  # jempol harus cukup jauh dari pangkal telunjuk
TERANG_MIN = 40  # rata-rata kecerahan ROI minimal (0..255)
PUTIH_MAKS = 0.4  # bagian ROI yang putih total (>= 250) maksimal 40%
TAJAM_MIN = 20.0  # ketajaman minimal (variansi Laplacian setelah CLAHE)

# (ujung jari, ruas tengah) untuk telunjuk, tengah, manis, kelingking
PASANGAN_JARI = [(8, 6), (12, 10), (16, 14), (20, 18)]


def sisi_dari_label(label):
    """Ubah label MediaPipe menjadi "kanan" atau "kiri".

    MediaPipe versi Python mengira fotonya dicerminkan (seperti kamera selfie), padahal foto dari
    browser tidak dicerminkan. Akibatnya tangan kanan sungguhan dilaporkan sebagai "Left".
    """
    return "kanan" if label == "Left" else "kiri"


def jari_terbuka(titik, lebar, tinggi):
    """True kalau kelima jari terbuka."""
    pergelangan = ke_piksel(titik, 0, lebar, tinggi)
    for ujung, ruas in PASANGAN_JARI:
        jarak_ujung = np.linalg.norm(ke_piksel(titik, ujung, lebar, tinggi) - pergelangan)
        jarak_ruas = np.linalg.norm(ke_piksel(titik, ruas, lebar, tinggi) - pergelangan)
        if jarak_ujung <= RASIO_JARI * jarak_ruas:
            return False

    jempol = np.linalg.norm(
        ke_piksel(titik, 4, lebar, tinggi) - ke_piksel(titik, 5, lebar, tinggi)
    )
    lebar_telapak = np.linalg.norm(
        ke_piksel(titik, 5, lebar, tinggi) - ke_piksel(titik, 17, lebar, tinggi)
    )
    return jempol > RASIO_JEMPOL * lebar_telapak


def menghadap_kamera(titik, sisi, lebar, tinggi):
    """True kalau telapak (bukan punggung tangan) yang menghadap kamera.

    Dihitung dari perkalian silang vektor pergelangan->telunjuk dan pergelangan->kelingking.
    Tandanya (positif/negatif) berbeda untuk telapak dan punggung, dan terbalik untuk tangan kiri.
    """
    p0 = ke_piksel(titik, 0, lebar, tinggi)
    a = ke_piksel(titik, 5, lebar, tinggi) - p0
    b = ke_piksel(titik, 17, lebar, tinggi) - p0
    silang = a[0] * b[1] - a[1] * b[0]
    return silang < 0 if sisi == "kanan" else silang > 0


def cek_kualitas_roi(roi):
    """Kembalikan None kalau ROI bisa dipakai, atau alasan penolakannya."""
    if roi.mean() < TERANG_MIN:
        return "Foto terlalu gelap, tambah cahaya"
    if (roi >= 250).mean() > PUTIH_MAKS:
        return "Foto terlalu terang"
    if cv2.Laplacian(pertegas_kontras(roi), cv2.CV_64F).var() < TAJAM_MIN:
        return "Foto kurang jelas, tahan tangan lebih diam"
    return None
