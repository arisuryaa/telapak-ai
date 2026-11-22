import base64
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import create_app
from conftest import PengenalPalsu
from steps import ke_data_url

FIX = Path(__file__).parent / "fixtures"
DATA = json.loads((FIX / "landmarks_test.json").read_text())


def lm(nama):
    return [SimpleNamespace(x=x / DATA["lebar"], y=y / DATA["tinggi"], z=0.0) for x, y in DATA[nama]]


def frame_uji():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


FRAME = ke_data_url(frame_uji(), ".jpg")
ESTADO = {"tangan": "terbuka_kanan"}  # which fixture the fake detector returns


def detektor_palsu(frame):
    if ESTADO["tangan"] is None:
        return None
    return lm(ESTADO["tangan"]), "Left"


@pytest.fixture
def client():
    ESTADO["tangan"] = "terbuka_kanan"
    app = create_app("sqlite://", deteksi=detektor_palsu, pengenal=PengenalPalsu())
    return app.test_client()


def sampel_valid(client):
    s = client.post("/api/sampel", json={"frame": FRAME}).get_json()
    assert s["valid"]
    return s


def body_daftar(kode, **ubah):
    body = {"nama": "dewa", "umur": 21, "status": "mahasiswa", "sisi": "kanan",
            "persetujuan": True, "kode": [kode] * 8, "sisi_terdeteksi": ["kanan"] * 8}
    body.update(ubah)
    return body


def test_scan_galeri_kosong(client):
    r = client.post("/api/scan", json={"frame": FRAME, "batas": 0.38}).get_json()
    assert r["hasil"]["status"] == "tidak_dikenal"
    assert r["hasil"]["pesan"] == "Belum ada yang terdaftar"
    assert [l["no"] for l in r["langkah"]][:7] == [1, 2, 3, 4, 5, 6, 7]


def test_daftar_lalu_dikenali(client):
    s = sampel_valid(client)
    assert s["sisi"] == "kanan"
    assert client.post("/api/orang", json=body_daftar(s["kode"])).status_code == 201
    r = client.post("/api/scan", json={"frame": FRAME, "batas": 0.38}).get_json()
    assert r["hasil"]["status"] == "dikenali" and r["hasil"]["nama"] == "dewa"
    assert r["hasil"]["jarak"] == pytest.approx(0.0, abs=1e-6) and r["langkah"][-1]["no"] == 8


def test_kepal_tidak_pernah_dikenali(client):
    s = sampel_valid(client)
    client.post("/api/orang", json=body_daftar(s["kode"]))
    ESTADO["tangan"] = "kepal_kanan"
    r = client.post("/api/scan", json={"frame": FRAME, "batas": 0.38}).get_json()
    assert r["hasil"]["status"] == "gagal" and r["hasil"]["pesan"] == "Buka semua jari"
    assert client.post("/api/sampel", json={"frame": FRAME}).get_json()["valid"] is False


def test_simpan_orang_input_salah(client):
    s = sampel_valid(client)
    salah = [
        body_daftar(s["kode"], persetujuan=False),
        body_daftar(s["kode"], nama=""),
        body_daftar(s["kode"], umur="dua puluh"),
        body_daftar(s["kode"], umur=0),
        body_daftar(s["kode"], umur=121),
        body_daftar(s["kode"], status="presiden"),
        {**body_daftar(s["kode"]), "kode": [s["kode"]] * 7},
        body_daftar(s["kode"], sisi_terdeteksi=["kiri"] * 8),
    ]
    for body in salah:
        assert client.post("/api/orang", json=body).status_code == 400


def test_simpan_telapak_duplikat(client):
    s = sampel_valid(client)
    orang = client.post("/api/orang", json=body_daftar(s["kode"])).get_json()["orang"]
    r = client.post("/api/orang", json=body_daftar(s["kode"], orang_id=orang["id"]))
    assert r.status_code == 409 and r.get_json()["perlu_konfirmasi"] is True
    r = client.post("/api/orang", json=body_daftar(s["kode"], orang_id=orang["id"], ganti=True))
    assert r.status_code == 201
    assert client.get(f"/api/orang/{orang['id']}").get_json()["tangan"] == ["kanan"]


def test_frame_rusak(client):
    assert client.post("/api/scan", json={"frame": "xx"}).status_code == 400


def test_hapus(client):
    s = sampel_valid(client)
    orang = client.post("/api/orang", json=body_daftar(s["kode"])).get_json()["orang"]
    assert client.delete(f"/api/orang/{orang['id']}").status_code == 204
    assert client.get(f"/api/orang/{orang['id']}").status_code == 404
    client.post("/api/orang", json=body_daftar(s["kode"]))
    assert client.delete("/api/orang").get_json() == {"dihapus": 1}


def test_api_mencatat_log(client, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="telapak"):
        s = sampel_valid(client)
        client.post("/api/orang", json=body_daftar(s["kode"]))
        client.post("/api/scan", json={"frame": FRAME, "batas": 0.38})
        client.post("/api/orang", json=body_daftar(s["kode"], umur=0))
    assert "sampel: diterima, sisi=kanan" in caplog.text
    assert "tersimpan: orang_id=1 sisi=kanan" in caplog.text
    # what went into the database: size and model version, then one line per sample
    assert "model=palsu-1, 4096 byte = 8 sampel x 128 angka" in caplog.text
    assert caplog.text.count("isi sampel ") == 8 and "panjang 1.000" in caplog.text
    assert "scan: dikenali, terdekat dewa (kanan) jarak=0.000" in caplog.text
    # every hand the scan was compared with, with its distance to each of the 8 samples
    assert "dibandingkan: dewa (kanan) jarak=0.000, per sampel: 0.000 0.000" in caplog.text
    assert "POST /api/orang -> 400: Umur harus angka" in caplog.text


def test_sampel_berisi_512_byte_float32(client):
    s = sampel_valid(client)
    v = np.frombuffer(base64.b64decode(s["kode"]), np.float32)
    assert v.shape == (128,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


def test_sampel_rusak_ditolak(client):
    sampel_valid(client)
    pendek = base64.b64encode(bytes(511)).decode("ascii")
    nan = base64.b64encode(np.full(128, np.nan, np.float32).tobytes()).decode("ascii")
    for rusak in (pendek, nan):
        r = client.post("/api/orang", json=body_daftar(rusak))
        assert r.status_code == 400 and r.get_json()["pesan"] == "Data sampel tidak lengkap"


def test_versi_model_lain_tidak_dibandingkan(tmp_path):
    url = f"sqlite:///{tmp_path / 'versi.db'}"
    lama = create_app(url, deteksi=detektor_palsu, pengenal=PengenalPalsu(versi="lama")).test_client()
    s = sampel_valid(lama)
    assert lama.post("/api/orang", json=body_daftar(s["kode"])).status_code == 201
    baru = create_app(url, deteksi=detektor_palsu, pengenal=PengenalPalsu(versi="baru")).test_client()
    r = baru.post("/api/scan", json={"frame": FRAME}).get_json()
    assert r["hasil"]["status"] == "tidak_dikenal" and r["jarak"] == []
    assert r["hasil"]["pesan"] == "Belum ada yang terdaftar dengan model ini (versi baru). Daftar ulang dulu."
    assert baru.get("/api/orang").get_json()[0]["versi_tangan"] == {"kanan": "lama"}


def test_pengaturan_model(client):
    p = client.get("/api/pengaturan").get_json()
    assert (p["metode"], p["model_versi"], p["batas_default"], p["roi_skala"]) == ("cnn", "palsu-1", 0.4, 1.0)
    assert (p["batas_min"], p["batas_maks"]) == (0.05, 1.0) and p["metode_nama"] == "Model CNN (ResNet18)"


def test_batas_dijepit_dan_bawaan_dari_model(client):
    assert client.post("/api/scan", json={"frame": FRAME, "batas": 3}).get_json()["batas"] == 1.0
    assert client.post("/api/scan", json={"frame": FRAME, "batas": 0.0}).get_json()["batas"] == 0.05
    assert client.post("/api/scan", json={"frame": FRAME}).get_json()["batas"] == 0.4


def test_batas_cadangan_jika_model_tanpa_batas():
    c = create_app("sqlite://", deteksi=detektor_palsu, pengenal=PengenalPalsu(batas=None)).test_client()
    assert c.get("/api/pengaturan").get_json()["batas_default"] == 0.4


def test_langkah_model(client):
    s = sampel_valid(client)
    client.post("/api/orang", json=body_daftar(s["kode"]))
    r = client.post("/api/scan", json={"frame": FRAME}).get_json()
    langkah = {l["no"]: l for l in r["langkah"]}
    assert [l["no"] for l in r["langkah"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [langkah[n]["judul"] for n in (5, 6, 7)] == ["Masukan model", "Model CNN: peta fitur", "Embedding 128 angka"]
    assert len(langkah[6]["gambar"]) == 8 and len(langkah[7]["gambar"]) == 1
    assert langkah[7]["angka"]["versi model"] == "palsu-1" and r["skala_maks"] == 1.0 and r["metode"] == "cnn"


def test_tanpa_file_model_pesan_jelas(monkeypatch, tmp_path):
    import app as app_mod
    monkeypatch.setattr(app_mod, "MODEL_PATH", tmp_path / "tidak_ada.pt")
    with pytest.raises(FileNotFoundError, match="Jalankan training dulu"):
        create_app("sqlite://", deteksi=detektor_palsu)
