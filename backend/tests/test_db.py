import pytest
from sqlalchemy.orm import Session

import db

EMBEDDING = [[0.1, -0.2, 0.3]] * 8


@pytest.fixture
def s():
    engine = db.buat_engine("sqlite://")
    with Session(engine) as sesi:
        yield sesi


def test_simpan_lalu_baca(s):
    orang = db.tambah_orang(s, "dewa", 21, "mahasiswa")
    db.simpan_tangan(s, orang.id, "kanan", EMBEDDING, "model-a")
    s.commit()

    semua = db.semua_tangan(s, "model-a")
    assert len(semua) == 1
    assert semua[0]["nama"] == "dewa" and semua[0]["embedding"] == EMBEDDING
    assert db.semua_tangan(s, "model-b") == []


def test_simpan_lagi_mengganti_data_lama(s):
    orang = db.tambah_orang(s, "dewa", 21, "mahasiswa")
    db.simpan_tangan(s, orang.id, "kanan", EMBEDDING, "model-a")
    db.simpan_tangan(s, orang.id, "kanan", [[1.0]] * 8, "model-b")
    s.commit()
    assert db.ambil_orang(s, orang.id)["tangan"] == [{"sisi": "kanan", "model": "model-b"}]


def test_hapus(s):
    a = db.tambah_orang(s, "ayu", 20, "dosen")
    db.tambah_orang(s, "budi", 22, "staf")
    db.simpan_tangan(s, a.id, "kiri", EMBEDDING, "model-a")
    s.commit()

    assert db.hapus_orang(s, a.id) is True
    assert db.hapus_orang(s, 999) is False
    s.commit()
    assert db.semua_tangan(s, "model-a") == []  # tangan ikut terhapus
    assert db.hapus_semua(s) == 1


def test_mysql_mati_pesan_jelas():
    with pytest.raises(db.DatabaseTidakTersedia, match="Nyalakan MySQL"):
        db.buat_engine("mysql+pymysql://root:@127.0.0.1:1/tidak_ada")
