import pytest

import model
from app import create_app
from conftest import FOTO


@pytest.fixture
def client(tangan):
    return create_app("sqlite://").test_client()


def ambil_sampel(client):
    s = client.post("/api/sampel", json={"foto": FOTO}).get_json()
    assert s["valid"], s["pesan"]
    return s


def body_daftar(embedding, **ubah):
    body = {
        "nama": "dewa",
        "umur": 21,
        "status": "mahasiswa",
        "sisi": "kanan",
        "persetujuan": True,
        "embedding": [embedding] * 8,
    }
    body.update(ubah)
    return body


def test_tanpa_model_berhenti_setelah_roi(client):
    p = client.get("/api/pengaturan").get_json()
    assert p["model"] is None and p["model_siap"] is False

    r = client.post("/api/scan", json={"foto": FOTO}).get_json()
    assert r["status"] == "gagal" and r["pesan"].startswith("Model belum dibuat")
    judul = [l["judul"] for l in r["langkah"]]
    assert judul == ["Foto dari kamera", "Deteksi tangan", "Telapak diluruskan", "ROI abu-abu"]

    s = client.post("/api/sampel", json={"foto": FOTO}).get_json()
    assert s["valid"] is False and s["sisi"] == "kanan"


def test_daftar_lalu_dikenali(client, model_palsu):
    s = ambil_sampel(client)
    assert s["sisi"] == "kanan" and len(s["embedding"]) == 128
    assert client.post("/api/orang", json=body_daftar(s["embedding"])).status_code == 201

    r = client.post("/api/scan", json={"foto": FOTO, "batas": 0.4}).get_json()
    assert r["status"] == "dikenali" and r["pesan"] == "Dikenali sebagai dewa"
    assert r["terdekat"]["jarak"] == pytest.approx(0.0, abs=1e-6)
    assert r["langkah"][-1]["judul"] == "Pencocokan" and r["langkah"][-1]["no"] == 6


def test_scan_tanpa_data_terdaftar(client, model_palsu):
    r = client.post("/api/scan", json={"foto": FOTO}).get_json()
    assert r["status"] == "tidak_dikenal" and r["jarak"] == []


def test_tangan_mengepal_ditolak(client, tangan, model_palsu):
    tangan["nama"] = "kepal_kanan"
    r = client.post("/api/scan", json={"foto": FOTO}).get_json()
    assert r["status"] == "gagal" and r["pesan"] == "Buka semua jari"


def test_input_daftar_salah(client, model_palsu):
    e = ambil_sampel(client)["embedding"]
    salah = [
        body_daftar(e, persetujuan=False),
        body_daftar(e, nama=""),
        body_daftar(e, umur="dua puluh"),
        body_daftar(e, umur=0),
        body_daftar(e, status="presiden"),
        body_daftar(e, sisi="tengah"),
        {**body_daftar(e), "embedding": [e] * 7},
        {**body_daftar(e), "embedding": [e] * 7 + [e[:10]]},
        {**body_daftar(e), "embedding": [["satu"]] * 8},
    ]
    for body in salah:
        assert client.post("/api/orang", json=body).status_code == 400


def test_tangan_sudah_terdaftar_perlu_konfirmasi(client, model_palsu):
    e = ambil_sampel(client)["embedding"]
    orang = client.post("/api/orang", json=body_daftar(e)).get_json()["orang"]

    r = client.post("/api/orang", json=body_daftar(e, orang_id=orang["id"]))
    assert r.status_code == 409 and r.get_json()["perlu_konfirmasi"] is True

    r = client.post("/api/orang", json=body_daftar(e, orang_id=orang["id"], ganti=True))
    assert r.status_code == 201
    assert r.get_json()["orang"]["tangan"] == [{"sisi": "kanan", "model": "palsu"}]


def test_foto_rusak(client):
    assert client.post("/api/scan", json={"foto": "bukan foto"}).status_code == 400
    assert client.post("/api/sampel", json={}).status_code == 400


def test_hapus(client, model_palsu):
    e = ambil_sampel(client)["embedding"]
    orang = client.post("/api/orang", json=body_daftar(e)).get_json()["orang"]
    assert client.delete(f"/api/orang/{orang['id']}").status_code == 204
    assert client.get(f"/api/orang/{orang['id']}").status_code == 404

    client.post("/api/orang", json=body_daftar(e))
    assert client.delete("/api/orang").get_json() == {"dihapus": 1}
    assert client.get("/api/orang").get_json() == []


def test_model_lain_tidak_dibandingkan(client, model_palsu, monkeypatch):
    e = ambil_sampel(client)["embedding"]
    client.post("/api/orang", json=body_daftar(e))
    monkeypatch.setattr(model, "NAMA_MODEL", "model-baru")
    r = client.post("/api/scan", json={"foto": FOTO}).get_json()
    assert r["status"] == "tidak_dikenal" and r["jarak"] == []


def test_batas(client, model_palsu, monkeypatch):
    assert client.post("/api/scan", json={"foto": FOTO, "batas": 3}).get_json()["batas"] == 1.0
    assert client.post("/api/scan", json={"foto": FOTO, "batas": 0}).get_json()["batas"] == 0.05

    monkeypatch.setenv("BATAS", "0.3")
    c = create_app("sqlite://").test_client()
    assert c.get("/api/pengaturan").get_json()["batas_bawaan"] == 0.3
