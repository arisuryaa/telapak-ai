from types import SimpleNamespace

import numpy as np

import palm


def titik_dari_piksel(piksel, lebar=640, tinggi=480):
    """21 titik tangan; titik yang tidak disebut disamakan dengan titik 0."""
    titik = [SimpleNamespace(x=piksel[0][0] / lebar, y=piksel[0][1] / tinggi) for _ in range(21)]
    for nomor, (x, y) in piksel.items():
        titik[nomor] = SimpleNamespace(x=x / lebar, y=y / tinggi)
    return titik


FOTO_HITAM = np.zeros((480, 640, 3), np.uint8)


def test_roi_berukuran_128():
    titik = titik_dari_piksel({0: (320, 420), 5: (360, 300), 17: (272, 312)})
    roi = palm.potong_roi(FOTO_HITAM, titik)
    assert roi["roi"].shape == (128, 128)
    assert roi["foto_lurus"].shape == FOTO_HITAM.shape


def test_kotak_selebar_telapak():
    titik = titik_dari_piksel({0: (320, 420), 5: (360, 300), 17: (272, 312)})
    x1, y1, x2, y2 = palm.potong_roi(FOTO_HITAM, titik)["kotak"]
    lebar_telapak = np.hypot(360 - 272, 300 - 312)
    assert abs((x2 - x1) - lebar_telapak) <= 1
    assert abs((y2 - y1) - lebar_telapak) <= 1


def test_tangan_terlalu_kecil():
    titik = titik_dari_piksel({0: (320, 420), 5: (330, 300), 17: (310, 300)})
    assert palm.potong_roi(FOTO_HITAM, titik) is None


def test_kotak_keluar_foto():
    # Tangan terlalu dekat ke kamera: kotak selebar telapak (400 piksel) tidak muat di foto.
    titik = titik_dari_piksel({0: (300, 420), 5: (500, 300), 17: (100, 312)})
    assert palm.potong_roi(FOTO_HITAM, titik) is None
