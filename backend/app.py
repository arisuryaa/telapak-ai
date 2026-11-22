"""app.py - Flask API for scanning, collecting samples, and managing enrolled people."""
import base64
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

import palm
from db import (SISI, STATUS, DatabaseTidakTersedia, Orang, TelapakSudahAda, ambil_orang,
                buat_engine, daftar_orang, hapus_orang, hapus_semua, muat_galeri,
                simpan_telapak, tambah_orang)
from pipeline import analisis, baca_frame
from steps import susun_langkah

BYTE_PER_SAMPEL = 128 * 4  # one sample is an embedding of 128 float32 numbers
JUMLAH_SAMPEL = 8
DATABASE_DEFAULT = "mysql+pymysql://root:@localhost:3306/telapak_ai"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "telapak.pt"
# Cosine distance of unit vectors is 0..2; in practice the same palm sits well below 1.
BATAS_CADANGAN, BATAS_MIN, BATAS_MAKS, SKALA_JARAK_MAKS = 0.4, 0.05, 1.0, 1.0
METODE_NAMA = "Model CNN (ResNet18)"

log = logging.getLogger("telapak")


def _galat(pesan, status=400, **extra):
    log.warning("%s %s -> %d: %s", request.method, request.path, status, pesan)
    return jsonify({"pesan": pesan, **extra}), status


def _batas(nilai, bawaan):
    """The threshold from the request, kept inside the slider range."""
    try:
        b = float(nilai)
    except (TypeError, ValueError):
        return bawaan
    return min(BATAS_MAKS, max(BATAS_MIN, b))


def _isi_sampel(e):
    """One embedding in words for the log: its first numbers and its length (always 1)."""
    awal = " ".join(f"{v:+.3f}" for v in e[:5])
    return f"5 angka pertama {awal} ..., panjang {float(np.linalg.norm(e)):.3f}"


def _decode_sampel(item):
    try:
        b = base64.b64decode(item, validate=True)
    except (ValueError, TypeError):
        return None
    if len(b) != BYTE_PER_SAMPEL:
        return None
    arr = np.frombuffer(b, np.float32).copy()
    return arr if np.isfinite(arr).all() else None


def jarak_per_sampel(kode, sampel):
    """Cosine distance from one embedding to each enrolled embedding of a hand (0 = same direction)."""
    return np.clip(1.0 - np.asarray(sampel, np.float32) @ np.asarray(kode, np.float32), 0.0, 2.0)


def create_app(database_url=None, deteksi=palm.deteksi_tangan, pengenal=None):
    """`pengenal` is the trained model (model.Pengenal); by default it is loaded from model/telapak.pt."""
    if database_url is None:
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        database_url = os.environ.get("DATABASE_URL", DATABASE_DEFAULT)
    if pengenal is None:
        from model import Pengenal  # torch is only loaded when the real model is used
        pengenal = Pengenal.dari_file(MODEL_PATH)
    batas_bawaan = BATAS_CADANGAN if pengenal.batas is None else round(float(pengenal.batas), 4)
    versi = pengenal.versi

    app = Flask(__name__)
    engine = buat_engine(database_url)
    log.info("database: %s", make_url(database_url).render_as_string(hide_password=True))
    log.info("model: %s versi %s (batas bawaan %.3f)", METODE_NAMA, versi, batas_bawaan)

    @app.get("/api/pengaturan")
    def pengaturan():
        return jsonify({"roi_skala": 1.0, "metode": "cnn", "metode_nama": METODE_NAMA, "model_versi": versi,
                        "batas_default": batas_bawaan, "batas_min": BATAS_MIN, "batas_maks": BATAS_MAKS,
                        "skala_jarak_maks": SKALA_JARAK_MAKS})

    @app.post("/api/scan")
    def scan():
        body = request.get_json(silent=True) or {}
        frame = baca_frame(body.get("frame", ""))
        if frame is None:
            return _galat("Gambar tidak valid")
        batas = _batas(body.get("batas"), batas_bawaan)
        info = {"metode": "cnn", "metode_nama": METODE_NAMA, "model_versi": versi, "skala_maks": SKALA_JARAK_MAKS}

        a = analisis(frame, pengenal, deteksi=deteksi)
        if a.gagal:
            return jsonify({
                "langkah": susun_langkah(a),
                "jarak": [],
                "batas": batas,
                "hasil": {"status": "gagal", "pesan": a.gagal},
                **info,
            })

        with Session(engine) as s:
            semua = muat_galeri(s)
        # Embeddings of another training of the model are not comparable, so leave them out.
        galeri = [g for g in semua if g["model_versi"] == versi]
        if not galeri:
            jarak = []
            pesan = ("Belum ada yang terdaftar" if not semua else
                     f"Belum ada yang terdaftar dengan model ini (versi {versi}). Daftar ulang dulu.")
            hasil = {"status": "tidak_dikenal", "pesan": pesan}
        else:
            jarak = []
            for g in galeri:
                per_sampel = jarak_per_sampel(a.kode, g["kode"])
                # a hand's distance is its closest sample, so one good enrolment photo is enough
                jarak.append({"telapak_id": g["telapak_id"], "orang_id": g["orang_id"], "nama": g["nama"],
                              "sisi": g["sisi"], "jarak": float(min(per_sampel))})
                log.info("dibandingkan: %s (%s) jarak=%.3f, per sampel: %s", g["nama"], g["sisi"], min(per_sampel),
                         " ".join(f"{d:.3f}" for d in per_sampel))
            jarak.sort(key=lambda d: d["jarak"])
            terdekat = jarak[0]
            if terdekat["jarak"] < batas:
                hasil = {"status": "dikenali", "pesan": f"Dikenali sebagai {terdekat['nama']}",
                         "orang_id": terdekat["orang_id"], "nama": terdekat["nama"],
                         "sisi": terdekat["sisi"], "jarak": terdekat["jarak"]}
            else:
                hasil = {"status": "tidak_dikenal", "pesan": "Tidak dikenal",
                         "jarak": terdekat["jarak"]}
        if jarak:
            log.info("scan: %s, terdekat %s (%s) jarak=%.3f, batas=%.2f", hasil["status"],
                     jarak[0]["nama"], jarak[0]["sisi"], jarak[0]["jarak"], batas)
        else:
            log.info("scan: %s (%s)", hasil["status"], hasil["pesan"])
        return jsonify({"langkah": susun_langkah(a, jarak=jarak, batas=batas),
                        "jarak": jarak, "batas": batas, "hasil": hasil, **info})

    @app.post("/api/sampel")
    def sampel():
        body = request.get_json(silent=True) or {}
        frame = baca_frame(body.get("frame", ""))
        if frame is None:
            return _galat("Gambar tidak valid")
        a = analisis(frame, pengenal, deteksi=deteksi)
        if a.gagal:
            log.info("sampel: ditolak, %s", a.gagal)
            return jsonify({"valid": False, "pesan": a.gagal, "sisi": None, "kode": None})
        log.info("sampel: diterima, sisi=%s", a.sisi)
        return jsonify({"valid": True, "pesan": "Sampel diterima", "sisi": a.sisi,
                        "kode": base64.b64encode(np.asarray(a.kode, np.float32).tobytes()).decode("ascii")})

    @app.post("/api/orang")
    def simpan_orang():
        body = request.get_json(silent=True) or {}
        if body.get("persetujuan") is not True:
            return _galat("Pendaftaran butuh persetujuan")
        sisi = body.get("sisi")
        if sisi not in SISI:
            return _galat("Sisi tangan harus kiri atau kanan")

        orang_id = body.get("orang_id")
        if orang_id is None:
            nama = str(body.get("nama", "")).strip()
            umur = body.get("umur")
            if not nama or len(nama) > 100:
                return _galat("Nama wajib diisi (maks. 100 karakter)")
            if not (isinstance(umur, int) and not isinstance(umur, bool) and 1 <= umur <= 120):
                return _galat("Umur harus angka 1–120")
            if body.get("status") not in STATUS:
                return _galat("Status tidak dikenal")

        daftar_item = body.get("kode")
        if not isinstance(daftar_item, list) or len(daftar_item) != JUMLAH_SAMPEL:
            return _galat("Data sampel tidak lengkap")
        sampel_arr = [_decode_sampel(item) for item in daftar_item]
        if any(x is None for x in sampel_arr):
            return _galat("Data sampel tidak lengkap")
        kode = np.stack(sampel_arr)

        terdeteksi = body.get("sisi_terdeteksi")
        if not isinstance(terdeteksi, list) or not terdeteksi:
            return _galat("Data sampel tidak lengkap")
        mayoritas = Counter(terdeteksi).most_common(1)[0][0]
        if mayoritas != sisi:
            return _galat(f"Tangan yang terdeteksi adalah tangan {mayoritas}, bukan {sisi}")

        ganti = body.get("ganti") is True
        with Session(engine) as s:
            if orang_id is None:
                orang_id = tambah_orang(s, nama, umur, body["status"]).id
            elif s.get(Orang, orang_id) is None:
                return _galat("Orang tidak ditemukan", 404)
            try:
                simpan_telapak(s, orang_id, sisi, kode, ganti=ganti, model_versi=versi)
            except TelapakSudahAda:
                s.rollback()
                return _galat(f"Tangan {sisi} sudah terdaftar. Ganti data lama?", 409,
                              perlu_konfirmasi=True)
            s.commit()
            log.info("tersimpan: orang_id=%s sisi=%s ganti=%s model=%s, %d byte = %d sampel x 128 angka",
                     orang_id, sisi, ganti, versi, kode.nbytes, len(kode))
            for i, k in enumerate(kode, 1):
                log.info("  isi sampel %d: %s", i, _isi_sampel(k))
            return jsonify({"orang": ambil_orang(s, orang_id)}), 201

    @app.get("/api/orang")
    def daftar():
        with Session(engine) as s:
            return jsonify(daftar_orang(s))

    @app.get("/api/orang/<int:orang_id>")
    def detail(orang_id):
        with Session(engine) as s:
            o = ambil_orang(s, orang_id)
        return (jsonify(o), 200) if o else _galat("Orang tidak ditemukan", 404)

    @app.delete("/api/orang/<int:orang_id>")
    def hapus(orang_id):
        with Session(engine) as s:
            ada = hapus_orang(s, orang_id)
            s.commit()
        if ada:
            log.info("dihapus: orang_id=%s", orang_id)
        return ("", 204) if ada else _galat("Orang tidak ditemukan", 404)

    @app.delete("/api/orang")
    def hapus_banyak():
        with Session(engine) as s:
            jumlah = hapus_semua(s)
            s.commit()
        log.info("dihapus semua: %d orang", jumlah)
        return jsonify({"dihapus": jumlah})

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    try:
        app = create_app()
    except (DatabaseTidakTersedia, FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)
    app.run(port=5000)
