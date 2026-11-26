"""app.py - API Flask untuk Telapak AI.

Jalankan:  python app.py
Dokumentasi API (Swagger UI): http://127.0.0.1:5000/api/docs
"""

import math
import os
import sys

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from sqlalchemy.orm import Session

import db
import model
from pencocokan import hitung_jarak
from pipeline import baca_foto, proses_foto
from steps import susun_langkah

JUMLAH_SAMPEL = 8  # foto yang diambil saat mendaftarkan satu tangan
BATAS_MIN = 0.05
BATAS_MAKS = 1.0
DATABASE_BAWAAN = "mysql+pymysql://root:@localhost:3306/telapak_ai"
FOLDER_BACKEND = os.path.dirname(os.path.abspath(__file__))

# Halaman Swagger UI yang membaca openapi.yaml. Swagger UI dimuat dari CDN.
HALAMAN_DOCS = """<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <title>Telapak AI API</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
  <script>SwaggerUIBundle({ url: "/api/openapi.yaml", dom_id: "#swagger-ui" });</script>
</body>
</html>"""


def galat(pesan, status=400, **tambahan):
    """Balasan error: {"pesan": "..."} dengan kode status HTTP."""
    return jsonify({"pesan": pesan, **tambahan}), status


def baca_batas(nilai, bawaan):
    """Ambil batas dari permintaan, dijaga tetap di antara BATAS_MIN dan BATAS_MAKS."""
    try:
        batas = float(nilai)
    except (TypeError, ValueError):
        return bawaan
    return min(BATAS_MAKS, max(BATAS_MIN, batas))


def embedding_valid(embedding):
    """True kalau embedding berisi JUMLAH_SAMPEL daftar angka yang panjangnya sama."""
    if not isinstance(embedding, list) or len(embedding) != JUMLAH_SAMPEL:
        return False
    for sampel in embedding:
        if not isinstance(sampel, list) or len(sampel) == 0 or len(sampel) != len(embedding[0]):
            return False
        for angka in sampel:
            if isinstance(angka, bool) or not isinstance(angka, (int, float)):
                return False
            if not math.isfinite(angka):
                return False
    return True


def create_app(database_url=None):
    load_dotenv(os.path.join(FOLDER_BACKEND, ".env"))
    database_url = database_url or os.environ.get("DATABASE_URL", DATABASE_BAWAAN)
    batas_bawaan = baca_batas(os.environ.get("BATAS"), 0.5)

    app = Flask(__name__)
    engine = db.buat_engine(database_url)

    @app.get("/api/docs")
    def docs():
        return HALAMAN_DOCS

    @app.get("/api/openapi.yaml")
    def openapi():
        return send_from_directory(FOLDER_BACKEND, "openapi.yaml", mimetype="text/yaml")

    @app.get("/api/pengaturan")
    def pengaturan():
        return jsonify(
            {
                "model": model.NAMA_MODEL,
                "model_siap": model.model_siap(),
                "jumlah_sampel": JUMLAH_SAMPEL,
                "batas_bawaan": batas_bawaan,
                "batas_min": BATAS_MIN,
                "batas_maks": BATAS_MAKS,
            }
        )

    @app.post("/api/scan")
    def scan():
        body = request.get_json(silent=True) or {}
        foto = baca_foto(body.get("foto"))
        if foto is None:
            return galat("Foto tidak valid")
        batas = baca_batas(body.get("batas"), batas_bawaan)

        hasil = proses_foto(foto)
        if hasil["gagal"]:
            return jsonify(
                {
                    "status": "gagal",
                    "pesan": hasil["gagal"],
                    "terdekat": None,
                    "batas": batas,
                    "jarak": [],
                    "langkah": susun_langkah(hasil),
                }
            )

        with Session(engine) as s:
            terdaftar = db.semua_tangan(s, model.NAMA_MODEL)
        jarak = hitung_jarak(hasil["embedding"], terdaftar)
        terdekat = jarak[0] if jarak else None

        if terdekat is None:
            status, pesan = "tidak_dikenal", "Belum ada yang terdaftar dengan model ini"
        elif terdekat["jarak"] < batas:
            status, pesan = "dikenali", f"Dikenali sebagai {terdekat['nama']}"
        else:
            status, pesan = "tidak_dikenal", "Tidak dikenal"

        return jsonify(
            {
                "status": status,
                "pesan": pesan,
                "terdekat": terdekat,
                "batas": batas,
                "jarak": jarak,
                "langkah": susun_langkah(hasil, jarak, batas),
            }
        )

    @app.post("/api/sampel")
    def sampel():
        body = request.get_json(silent=True) or {}
        foto = baca_foto(body.get("foto"))
        if foto is None:
            return galat("Foto tidak valid")

        hasil = proses_foto(foto)
        if hasil["gagal"]:
            return jsonify({"valid": False, "pesan": hasil["gagal"], "sisi": hasil["sisi"], "embedding": None})
        return jsonify({"valid": True, "pesan": "Sampel diterima", "sisi": hasil["sisi"], "embedding": hasil["embedding"]})

    @app.post("/api/orang")
    def simpan_orang():
        body = request.get_json(silent=True) or {}
        if body.get("persetujuan") is not True:
            return galat("Pendaftaran butuh persetujuan")
        if not model.model_siap():
            return galat("Model belum dibuat")
        sisi = body.get("sisi")
        if sisi not in db.SISI:
            return galat("Sisi tangan harus kiri atau kanan")
        if not embedding_valid(body.get("embedding")):
            return galat(f"Data sampel tidak lengkap, butuh {JUMLAH_SAMPEL} embedding")

        orang_id = body.get("orang_id")
        if orang_id is None:
            nama = str(body.get("nama", "")).strip()
            umur = body.get("umur")
            if not nama or len(nama) > 100:
                return galat("Nama wajib diisi (maks. 100 karakter)")
            if isinstance(umur, bool) or not isinstance(umur, int) or not 1 <= umur <= 120:
                return galat("Umur harus angka 1–120")
            if body.get("status") not in db.STATUS:
                return galat("Status tidak dikenal")

        with Session(engine) as s:
            if orang_id is None:
                orang_id = db.tambah_orang(s, nama, umur, body["status"]).id
            elif s.get(db.Orang, orang_id) is None:
                return galat("Orang tidak ditemukan", 404)

            sudah_ada = db.cari_tangan(s, orang_id, sisi) is not None
            if sudah_ada and body.get("ganti") is not True:
                return galat(f"Tangan {sisi} sudah terdaftar. Ganti data lama?", 409, perlu_konfirmasi=True)

            db.simpan_tangan(s, orang_id, sisi, body["embedding"], model.NAMA_MODEL)
            s.commit()
            return jsonify({"orang": db.ambil_orang(s, orang_id)}), 201

    @app.get("/api/orang")
    def daftar():
        with Session(engine) as s:
            return jsonify(db.daftar_orang(s))

    @app.get("/api/orang/<int:orang_id>")
    def detail(orang_id):
        with Session(engine) as s:
            orang = db.ambil_orang(s, orang_id)
        if orang is None:
            return galat("Orang tidak ditemukan", 404)
        return jsonify(orang)

    @app.delete("/api/orang/<int:orang_id>")
    def hapus(orang_id):
        with Session(engine) as s:
            ada = db.hapus_orang(s, orang_id)
            s.commit()
        if not ada:
            return galat("Orang tidak ditemukan", 404)
        return "", 204

    @app.delete("/api/orang")
    def hapus_semua():
        with Session(engine) as s:
            jumlah = db.hapus_semua(s)
            s.commit()
        return jsonify({"dihapus": jumlah})

    return app


if __name__ == "__main__":
    try:
        app = create_app()
    except db.DatabaseTidakTersedia as e:
        print(e)
        sys.exit(1)
    app.run(port=5000)
