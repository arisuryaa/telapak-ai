import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import training  # noqa: F401  (puts backend/ on sys.path)

FIX = Path(__file__).resolve().parents[2] / "backend" / "tests" / "fixtures"
DATA = json.loads((FIX / "landmarks_test.json").read_text())


def _deteksi_tiruan(frame):
    """Always the same open right hand, whatever the photo shows."""
    titik = [SimpleNamespace(x=x / DATA["lebar"], y=y / DATA["tinggi"], z=0.0) for x, y in DATA["terbuka_kanan"]]
    return titik, "Left"


@pytest.fixture
def deteksi_tiruan():
    return _deteksi_tiruan


def _buat_foto_tongji(folder, sesi, nomor_list):
    """Textured 640x480 photos named like Tongji (00001.bmp ...) in folder/sesi/."""
    tujuan = Path(folder) / sesi
    tujuan.mkdir(parents=True, exist_ok=True)
    for n in nomor_list:
        img = np.random.default_rng(n).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(tujuan / f"{n:05d}.bmp"), cv2.GaussianBlur(img, (5, 5), 0))


@pytest.fixture
def buat_foto_tongji():
    return _buat_foto_tongji


def tulis_roi(folder, sesi, nomor_list, benih=0):
    """Random 128x128 grey ROI PNGs named like Tongji in folder/sesi/. Each palm gets its own base texture."""
    tujuan = Path(folder) / sesi
    tujuan.mkdir(parents=True, exist_ok=True)
    for n in nomor_list:
        telapak = (n - 1) // 10
        dasar = np.random.default_rng(benih * 1000 + telapak).integers(0, 256, (128, 128), dtype=np.uint8)
        noise = np.random.default_rng(n + 7).normal(0, 6, (128, 128))
        cv2.imwrite(str(tujuan / f"{n:05d}.png"), np.clip(dasar + noise, 0, 255).astype(np.uint8))


class PengenalPalsu:
    """Same interface as model.Pengenal, without torch: the embedding is the ROI shrunk to 8x16."""
    versi = "palsu-1"
    batas = 0.4

    def embedding_banyak(self, roi_list):
        e = np.stack([cv2.resize(r, (16, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for r in roi_list])
        e -= e.mean(axis=1, keepdims=True)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-6)

    def embedding(self, roi):
        return self.embedding_banyak([roi])[0]
