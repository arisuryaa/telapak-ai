import numpy as np
import pytest
import torch
import torch.nn.functional as F

import palm

from model import ArcFace, JaringanTelapak, Pengenal, siapkan_masukan, simpan_model


def test_embedding_panjang_satu():
    j = JaringanTelapak(pretrained=False).eval()
    e = j(siapkan_masukan([np.full((128, 128), v, np.uint8) for v in (30, 200)]))
    assert e.shape == (2, 128) and torch.allclose(e.norm(dim=1), torch.ones(2), atol=1e-5)


def test_masukan_tiga_kanal_ternormalisasi():
    x = siapkan_masukan([np.zeros((128, 128), np.uint8)])
    assert x.shape == (1, 3, 128, 128) and x.dtype == torch.float32
    # each channel is (CLAHE value / 255 - mean) / std of ImageNet, so channels differ only by those constants
    v = palm.terapkan_clahe(np.zeros((128, 128), np.uint8))[0, 0] / 255.0
    assert [x[0, c, 0, 0].item() for c in range(3)] == pytest.approx(
        [(v - m) / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))], abs=1e-4)


def test_arcface_bentuk_dan_margin():
    a = ArcFace(5)
    e = F.normalize(torch.randn(4, 128), dim=1)
    lab = torch.tensor([0, 1, 2, 3])
    keluar = a(e, lab)
    assert keluar.shape == (4, 5)
    tanpa_margin = 30.0 * F.linear(e, F.normalize(a.bobot, dim=1))
    assert (keluar.gather(1, lab[:, None]) < tanpa_margin.gather(1, lab[:, None])).all()
    # the other classes are not penalised
    lain = torch.ones_like(keluar, dtype=torch.bool).scatter_(1, lab[:, None], False)
    assert torch.allclose(keluar[lain], tanpa_margin[lain])


def test_simpan_muat_pengenal(tmp_path):
    j = JaringanTelapak(pretrained=False)
    simpan_model(tmp_path / "m.pt", j, versi="uji-1", batas=0.4, info={"epoch": 1})
    p = Pengenal.dari_file(tmp_path / "m.pt", perangkat="cpu")
    roi = np.random.default_rng(0).integers(0, 256, (128, 128), dtype=np.uint8)
    assert p.versi == "uji-1" and p.batas == 0.4
    e = p.embedding(roi)
    assert e.dtype == np.float32 and e.shape == (128,)
    np.testing.assert_allclose(e, p.embedding_banyak([roi])[0], atol=1e-5)
    d = p.detail(roi)
    assert d["masukan"].shape == (128, 128) and d["masukan"].dtype == np.uint8
    assert len(d["fitur"]) == 8 and d["fitur"][0].shape == (32, 32) and d["fitur"][0].dtype == np.uint8
    np.testing.assert_allclose(d["embedding"], e, atol=1e-5)


def test_model_tidak_ada_pesan_jelas(tmp_path):
    with pytest.raises(FileNotFoundError, match="Jalankan training dulu"):
        Pengenal.dari_file(tmp_path / "tidak_ada.pt")
