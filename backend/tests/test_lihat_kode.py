import cv2
import numpy as np
import pytest
from sqlalchemy.orm import Session

from db import buat_engine, simpan_telapak, tambah_orang
from test_bandingkan import buat_foto, deteksi_tiruan
from tools import lihat_kode as lk


def kode_contoh():
    """A tiny Canny-style code: one vertical line of direction 1 and one of direction 4, the rest no edge."""
    k = np.zeros((128, 128), np.uint8)
    k[40:90, 60] = 1
    k[60:100, 70] = 4
    return k


def test_potongan_angka_canny_memakai_titik_untuk_bukan_tepi():
    teks = lk.potongan_angka(kode_contoh(), "canny", y=62, x=56, tinggi=4, lebar=20)
    baris = teks.splitlines()
    assert baris[0] == "x=56..75" and len(baris) == 2 + 4
    assert baris[2] == "y=62  ....1.........4....."  # column 60 holds a 1 and column 70 a 4
    assert "0" not in baris[2].split()[-1]  # no zeros are printed: a no-edge pixel is a dot


def test_potongan_angka_gabor_mencetak_nol_apa_adanya():
    k = np.zeros((128, 128), np.uint8)
    k[10, 10] = 3
    assert lk.potongan_angka(k, "gabor", y=10, x=8, tinggi=1, lebar=6).splitlines()[2] == "y=10  003000"


def test_jendela_dijepit_ke_dalam_tabel():
    teks = lk.potongan_angka(kode_contoh(), "canny", y=500, x=500, tinggi=5, lebar=10)
    assert teks.splitlines()[0] == "x=118..127" and teks.splitlines()[2].startswith("y=123")


def test_ringkasan_kode():
    r = lk.ringkasan_kode(kode_contoh(), "canny")
    assert "tepi: 90 (0.5%)" in r and "50 piksel" in r and "40 piksel" in r
    assert "angka 1 = garis 0-30 derajat: hampir datar" in r and "angka 4 = garis 90-120 derajat: hampir tegak, turun ke kiri" in r
    assert "angka 0 = garis tegak" in lk.ringkasan_kode(kode_contoh(), "gabor")  # Gabor keeps its own meaning


def test_gambar_dan_lembar():
    g = lk.kode_ke_gambar(kode_contoh(), "canny", besar=2)
    assert g.shape == (256, 256, 3) and g[:, :, :].sum() > 0
    assert (g[0, 0] == 0).all()  # no edge stays black
    semua = np.stack([kode_contoh()] * 8)
    assert lk.lembar_sampel(semua, "canny").shape == (2 * (256 + 24), 4 * 256, 3)


@pytest.fixture
def db_tiruan(tmp_path):
    engine = buat_engine(f"sqlite:///{tmp_path / 'lihat.db'}")
    with Session(engine) as s:
        o = tambah_orang(s, "dewa", 21, "mahasiswa")
        simpan_telapak(s, o.id, "kanan", np.stack([kode_contoh()] * 8), roi_skala=1.0, metode="canny")
        s.commit()
    return engine


def test_daftar_dan_kode_dari_db(db_tiruan):
    assert lk.daftar_tangan(db_tiruan) == [(1, "dewa", "kanan", "canny", 1.0, 8 * 128 * 128)]
    info, semua = lk.kode_dari_db(db_tiruan, 1)
    assert info["nama"] == "dewa" and info["metode"] == "canny" and info["ukuran"] == 131072
    assert semua.shape == (8, 128, 128) and np.array_equal(semua[0], kode_contoh())
    with pytest.raises(ValueError, match="Tidak ada tangan dengan id 9"):
        lk.kode_dari_db(db_tiruan, 9)


def test_main_daftar_dan_id(db_tiruan, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(lk, "buka_engine", lambda: db_tiruan)
    assert lk.main(["--daftar"]) == 0
    assert "dewa" in capsys.readouterr().out
    assert lk.main(["--id", "1", "--sampel", "2", "--gambar", str(tmp_path / "gambar")]) == 0
    keluar = capsys.readouterr().out
    assert "Tangan kanan milik dewa (id 1), metode canny" in keluar and "131072 byte = 8 sampel" in keluar
    assert "y=44" in keluar and (tmp_path / "gambar" / "tangan_1.png").exists()
    assert lk.main(["--id", "1", "--sampel", "9"]) == 1
    assert lk.main(["--id", "7"]) == 1


@pytest.mark.parametrize("metode", ["canny", "gabor"])
def test_kode_dari_foto_memakai_pipeline_aplikasi(tmp_path, metode):
    buat_foto(tmp_path, "dewa_kanan", 1, 1)
    foto = tmp_path / "dewa_kanan" / "00.png"
    kode = lk.kode_dari_foto(foto, metode, 1.0, deteksi_tiruan)
    assert kode.shape == (128, 128) and kode.max() <= (6 if metode == "canny" else 5)
    cv2.imwrite(str(tmp_path / "gelap.png"), np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(ValueError, match="ditolak seperti di aplikasi: Foto terlalu gelap"):
        lk.kode_dari_foto(tmp_path / "gelap.png", metode, 1.0, deteksi_tiruan)
    with pytest.raises(ValueError, match="tidak bisa dibaca"):
        lk.kode_dari_foto(tmp_path / "tidak_ada.png", metode, 1.0, deteksi_tiruan)


def test_cocokkan_dengan_db(db_tiruan, monkeypatch, capsys, tmp_path):
    """--cocokkan runs a photo through the app and compares it with every saved hand, sample by sample."""
    buat_foto(tmp_path, "dewa_kanan", 1, 1)
    foto = tmp_path / "dewa_kanan" / "00.png"
    kode = lk.kode_dari_foto(foto, "canny", 1.0, deteksi_tiruan)
    hasil = lk.cocokkan(db_tiruan, kode, "canny", 1.0)
    assert len(hasil) == 1 and hasil[0]["nama"] == "dewa" and len(hasil[0]["per_sampel"]) == 8
    assert hasil[0]["jarak"] == min(hasil[0]["per_sampel"])
    assert lk.cocokkan(db_tiruan, kode, "gabor", 1.0) == []  # a hand saved with Canny is never compared with Gabor

    monkeypatch.setattr(lk, "buka_engine", lambda: db_tiruan)
    monkeypatch.setattr(lk.palm, "deteksi_tangan", deteksi_tiruan)
    assert lk.main(["--cocokkan", str(foto), "--metode", "canny", "--skala", "1"]) == 0
    keluar = capsys.readouterr().out
    assert "dewa (kanan, id 1)" in keluar and "sampel 8" in keluar and "batas 0.75" in keluar


def test_mysql_mati_pesan_jelas(monkeypatch, capsys):
    from sqlalchemy import create_engine
    monkeypatch.setattr(lk, "buka_engine", lambda: create_engine("mysql+pymysql://root:@127.0.0.1:1/tidak_ada"))
    assert lk.main(["--daftar"]) == 1
    assert "MySQL tidak bisa dihubungi" in capsys.readouterr().err
