import cv2
import numpy as np

from training.data import buat_pembagian, id_telapak, salin_roi_bawaan, siapkan_roi
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
    assert hasil == {"total": 1, "berhasil": 0, "gagal": ["session2/00001.bmp"]}


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
    p = buat_pembagian(tmp_path)
    semua = p["latih"] + p["validasi"] + p["uji"]
    assert len(semua) == len(set(semua)) == 20 and 20 not in semua and 21 not in semua
    assert (len(p["latih"]), len(p["validasi"]), len(p["uji"])) == (14, 2, 4)
    assert buat_pembagian(tmp_path) == p  # fixed seed = same split
