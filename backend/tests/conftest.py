"""Alat bantu tes: titik tangan dari file contoh, detektor tangan palsu, dan model palsu."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import model
import palm
from steps import ke_data_url

FIXTURES = Path(__file__).parent / "fixtures"
CONTOH = json.loads((FIXTURES / "landmarks_test.json").read_text())
LEBAR, TINGGI = CONTOH["lebar"], CONTOH["tinggi"]


def titik_tangan(nama):
    """21 titik tangan contoh, misalnya "terbuka_kanan", "kepal_kanan", "punggung_kanan"."""
    return [SimpleNamespace(x=x / LEBAR, y=y / TINGGI, z=0.0) for x, y in CONTOH[nama]]


def foto_uji():
    """Foto 640x480 berisi tekstur acak, cukup tajam untuk lolos cek kualitas ROI."""
    acak = np.random.default_rng(0).integers(0, 256, (TINGGI, LEBAR, 3), dtype=np.uint8)
    return cv2.GaussianBlur(acak, (5, 5), 0)


FOTO = ke_data_url(foto_uji())


@pytest.fixture
def tangan(monkeypatch):
    """Ganti MediaPipe dengan detektor palsu. Ubah tangan["nama"] untuk memilih tangan contoh."""
    pilihan = {"nama": "terbuka_kanan"}

    def deteksi_palsu(foto):
        if pilihan["nama"] is None:
            return None
        return titik_tangan(pilihan["nama"]), "Left"  # "Left" dari MediaPipe Python = tangan kanan

    monkeypatch.setattr(palm, "deteksi_tangan", deteksi_palsu)
    return pilihan


def embedding_palsu(roi):
    """Model palsu: ROI diperkecil menjadi 8x16 = 128 angka."""
    kecil = cv2.resize(roi, (16, 8), interpolation=cv2.INTER_AREA).astype(float).ravel()
    return list(kecil - kecil.mean())


@pytest.fixture
def model_palsu(monkeypatch):
    monkeypatch.setattr(model, "NAMA_MODEL", "palsu")
    monkeypatch.setattr(model, "buat_embedding", embedding_palsu)
