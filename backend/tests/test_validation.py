from pathlib import Path

import cv2
import numpy as np

from conftest import LEBAR, TINGGI, titik_tangan
from validation import cek_kualitas_roi, jari_terbuka, menghadap_kamera, sisi_dari_label

ROI_CONTOH = np.load(Path(__file__).parent / "fixtures" / "roi_test.npy")


def test_sisi_dari_label():
    assert sisi_dari_label("Left") == "kanan"
    assert sisi_dari_label("Right") == "kiri"


def test_jari_terbuka_dan_mengepal():
    assert jari_terbuka(titik_tangan("terbuka_kanan"), LEBAR, TINGGI)
    assert not jari_terbuka(titik_tangan("kepal_kanan"), LEBAR, TINGGI)


def test_telapak_atau_punggung():
    assert menghadap_kamera(titik_tangan("terbuka_kanan"), "kanan", LEBAR, TINGGI)
    assert not menghadap_kamera(titik_tangan("punggung_kanan"), "kanan", LEBAR, TINGGI)


def roi_telapak_halus():
    """Kulit halus dengan garis telapak samar, seperti ROI webcam sungguhan."""
    y, x = np.mgrid[0:128, 0:128].astype(np.float32)
    img = 175 + 8 * np.sin(x / 37) + 6 * np.cos(y / 29)
    for a, b, c in [(1, -0.6, -20), (0.3, -1, 60), (1, 0.9, -150), (-0.4, -1, 100)]:
        jarak = np.abs(a * x + b * y + c) / np.hypot(a, b)
        img -= 10 * np.exp(-(jarak**2) / 3)
    img += np.random.default_rng(3).normal(0, 0.8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_kualitas_roi():
    assert cek_kualitas_roi(ROI_CONTOH) is None
    assert cek_kualitas_roi(roi_telapak_halus()) is None
    assert cek_kualitas_roi(np.full((128, 128), 10, np.uint8)) == "Foto terlalu gelap, tambah cahaya"
    assert cek_kualitas_roi(np.full((128, 128), 252, np.uint8)) == "Foto terlalu terang"
    buram = cv2.GaussianBlur(roi_telapak_halus(), (0, 0), 3)
    assert cek_kualitas_roi(buram) == "Foto kurang jelas, tahan tangan lebih diam"
