import numpy as np

from model import Pengenal
from training.latih import DatasetTelapak, latih
from training.tests.conftest import tulis_roi


def _roi_kecil(folder, telapak_list):
    for t in telapak_list:
        tulis_roi(folder, "session1", range(10 * t + 1, 10 * t + 11))
        tulis_roi(folder, "session2", range(10 * t + 1, 10 * t + 3))


def test_dataset_label_berurutan(tmp_path):
    _roi_kecil(tmp_path, [7, 3])
    ds = DatasetTelapak(tmp_path, [7, 3], augmentasi=False)
    x, y = ds[0]
    assert x.shape == (3, 128, 128) and len(ds) == 24 and set(ds.label) == {0, 1}
    xa, _ = DatasetTelapak(tmp_path, [7, 3], augmentasi=True)[0]
    assert xa.shape == (3, 128, 128)


def test_latih_singkat_menyimpan_model_dan_riwayat(tmp_path, monkeypatch):
    _roi_kecil(tmp_path / "roi", range(6))
    pembagian = {"latih": [0, 1, 2, 3], "validasi": [4, 5], "uji": []}
    monkeypatch.chdir(tmp_path)
    riwayat = latih(tmp_path / "roi", pembagian, tmp_path / "m.pt", epoch=2, batch=8, pretrained=False, perangkat="cpu")
    assert [r["epoch"] for r in riwayat] == [1, 2] and all(np.isfinite(r["loss"]) for r in riwayat)
    assert all(0.0 <= r["eer_validasi"] <= 1.0 for r in riwayat)
    p = Pengenal.dari_file(tmp_path / "m.pt", perangkat="cpu")
    assert p.batas is not None and p.versi
    assert (tmp_path / "hasil" / "grafik_training.png").exists()
    assert (tmp_path / "hasil" / "riwayat_training.csv").read_text().startswith("epoch,loss,akurasi_latih,eer_validasi")
