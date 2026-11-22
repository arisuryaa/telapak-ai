import numpy as np
import pytest

from training.evaluasi import evaluasi
from training.metrik import eer, jarak_kosinus, roc, skor_identifikasi
from training.tests.conftest import PengenalPalsu, tulis_roi


def test_jarak_kosinus():
    v = np.eye(3, dtype=np.float32)
    d = jarak_kosinus(v[:2], v)
    assert d.shape == (2, 3) and d[0, 0] == pytest.approx(0.0) and d[0, 1] == pytest.approx(1.0)


def test_eer_terpisah_sempurna_dan_tumpang_tindih():
    e, b = eer(np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.7, 0.8]))
    assert e == 0.0 and 0.3 <= b < 0.6
    e2, _ = eer(np.array([0.1, 0.5]), np.array([0.4, 0.9]))
    assert e2 == pytest.approx(0.5)


def test_roc_naik_dari_nol_ke_satu():
    far, tar = roc(np.array([0.1, 0.2]), np.array([0.5, 0.6, 0.7]))
    assert far[0] == 0.0 and far[-1] == 1.0 and tar[-1] == 1.0
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0)


def test_skor_identifikasi_sampel_terdekat():
    v = np.eye(128, dtype=np.float32)
    daftar = {0: v[[0, 1]], 1: v[[2, 3]]}
    asli, palsu, rank1 = skor_identifikasi(daftar, [(0, v[1]), (1, v[0])])
    assert asli.tolist() == [0.0, 1.0] and sorted(palsu.tolist()) == [0.0, 1.0] and rank1 == 0.5


def test_evaluasi_melewati_telapak_tidak_lengkap(tmp_path):
    tulis_roi(tmp_path, "session1", range(1, 21))   # palms 0 and 1 complete
    tulis_roi(tmp_path, "session2", range(1, 21))
    tulis_roi(tmp_path, "session1", range(21, 26))  # palm 2: only 5 in session1
    tulis_roi(tmp_path, "session2", range(21, 31))
    hasil = evaluasi(PengenalPalsu(), tmp_path, [0, 1, 2])
    assert hasil["jumlah_telapak"] == 2 and hasil["jumlah_scan"] == 20
    assert 0.0 <= hasil["eer"] <= 1.0 and hasil["rank1"] == 1.0  # each palm has its own texture
