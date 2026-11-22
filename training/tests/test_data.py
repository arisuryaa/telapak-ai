import cv2
import numpy as np

import pytest

from training.data import buat_pembagian, id_telapak, potong_roi, salin_roi_bawaan, siapkan_roi
from training.tests.conftest import tulis_roi


def test_id_telapak():
    assert [id_telapak(n) for n in ["00001.bmp", "00010", "00011.jpg", "06000"]] == [0, 0, 1, 599]


def test_siapkan_roi_melewati_foto_gagal(tmp_path, deteksi_tiruan, buat_foto_tongji):
    buat_foto_tongji(tmp_path / "asli", "session1", range(1, 4))
    (tmp_path / "asli" / "session1" / "00004.bmp").write_bytes(b"rusak")
    hasil = siapkan_roi(tmp_path / "asli", tmp_path / "roi", deteksi=deteksi_tiruan)
    assert hasil["total"] == 4 and hasil["berhasil"] == 3 and hasil["gagal"] == ["session1/00004.bmp"]
    roi = cv2.imread(str(tmp_path / "roi" / "session1" / "00001.png"), cv2.IMREAD_GRAYSCALE)
    assert roi.shape == (128, 128)


def test_siapkan_roi_tangan_tidak_terdeteksi_dihitung_gagal(tmp_path, buat_foto_tongji):
    buat_foto_tongji(tmp_path / "asli", "session2", [1])
    hasil = siapkan_roi(tmp_path / "asli", tmp_path / "roi", deteksi=lambda f: None)
    assert hasil == {"total": 1, "berhasil": 0, "gagal": ["session2/00001.bmp"], "keluar_foto": 0}


def test_salin_roi_bawaan_jadi_abu_abu_128(tmp_path):
    (tmp_path / "bawaan" / "session1").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "bawaan" / "session1" / "00001.bmp"), np.full((200, 180, 3), 90, np.uint8))
    hasil = salin_roi_bawaan(tmp_path / "bawaan", tmp_path / "roi")
    roi = cv2.imread(str(tmp_path / "roi" / "session1" / "00001.png"), cv2.IMREAD_UNCHANGED)
    assert hasil["berhasil"] == 1 and roi.shape == (128, 128)


def test_pembagian_per_telapak_tidak_bocor(tmp_path):
    tulis_roi(tmp_path, "session1", range(1, 201))   # palms 0..19, 10 photos each
    tulis_roi(tmp_path, "session2", range(1, 201))
    tulis_roi(tmp_path, "session1", range(201, 206))  # palm 20: only 5 photos in session1
    tulis_roi(tmp_path, "session2", range(201, 211))
    tulis_roi(tmp_path, "session1", range(211, 221))  # palm 21: nothing in session2
    tulis_roi(tmp_path, "session2", [221])            # palm 22: a single ROI, too few to learn from
    p = buat_pembagian(tmp_path)
    semua = p["latih"] + p["validasi"] + p["uji"]
    assert len(semua) == len(set(semua)) == 22 and 22 not in semua
    # incomplete palms cannot be enrolled and scanned, but still teach the model: they go to training only
    assert 20 in p["latih"] and 21 in p["latih"]
    assert (len(p["latih"]), len(p["validasi"]), len(p["uji"])) == (16, 2, 4)
    assert buat_pembagian(tmp_path) == p  # fixed seed = same split


def _tangan_lurus(tinggi_foto):
    """A level hand in a 400 px wide photo: finger bases at y=300 (x 200 and 300), wrist below.

    The palm is 100 px wide, so the ROI box runs from y=315 to y=415. A photo of height 410 cuts
    5% of the box off, height 385 cuts 30%.
    """
    from types import SimpleNamespace
    titik = [SimpleNamespace(x=250 / 400, y=450 / tinggi_foto, z=0.0) for _ in range(21)]
    titik[5] = SimpleNamespace(x=200 / 400, y=300 / tinggi_foto, z=0.0)
    titik[17] = SimpleNamespace(x=300 / 400, y=300 / tinggi_foto, z=0.0)
    foto = np.random.default_rng(0).integers(0, 256, (tinggi_foto, 400, 3), dtype=np.uint8)
    return foto, titik


def test_kotak_sedikit_keluar_foto_tetap_dipotong():
    foto, titik = _tangan_lurus(410)
    roi, luar = potong_roi(foto, titik)
    assert roi.shape == (128, 128) and luar == pytest.approx(0.05, abs=0.02)
    assert (roi == 0).mean() < 0.01  # the missing strip is mirrored from the photo, not black


def test_kotak_banyak_keluar_foto_dilewati():
    foto, titik = _tangan_lurus(385)
    roi, luar = potong_roi(foto, titik)
    assert roi is None and luar == pytest.approx(0.30, abs=0.02)


def test_kotak_di_dalam_foto_tidak_ada_bagian_luar():
    foto, titik = _tangan_lurus(600)
    roi, luar = potong_roi(foto, titik)
    assert roi.shape == (128, 128) and luar == 0.0
    import palm
    assert np.array_equal(roi, palm.ambil_roi(foto, titik))  # same crop as the app when the box fits


def test_siapkan_roi_menghitung_kotak_keluar_foto(tmp_path, monkeypatch):
    import training.data as data_mod
    buat = tmp_path / "asli" / "session1"
    buat.mkdir(parents=True)
    foto_ok, titik_ok = _tangan_lurus(410)
    foto_luar, titik_luar = _tangan_lurus(385)
    cv2.imwrite(str(buat / "00001.png"), foto_ok)
    cv2.imwrite(str(buat / "00002.png"), foto_luar)
    hasil = siapkan_roi(tmp_path / "asli", tmp_path / "roi",
                        deteksi=lambda f: (titik_ok if f.shape[0] == 410 else titik_luar, "Left"))
    assert (hasil["berhasil"], hasil["gagal"], hasil["keluar_foto"]) == (1, ["session1/00002.png"], 1)
