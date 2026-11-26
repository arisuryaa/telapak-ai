"""db.py - tabel MySQL dan fungsi untuk membaca/menulis data.

Ada dua tabel:
- orang  : nama, umur, status
- tangan : satu tangan (kiri/kanan) milik seseorang, berisi beberapa embedding dari model

Foto tidak pernah disimpan, hanya embedding (daftar angka).
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

STATUS = ("mahasiswa", "dosen", "staf", "tamu")
SISI = ("kiri", "kanan")


class DatabaseTidakTersedia(Exception):
    """MySQL tidak bisa dihubungi."""


class Base(DeclarativeBase):
    pass


class Orang(Base):
    __tablename__ = "orang"

    id: Mapped[int] = mapped_column(primary_key=True)
    nama: Mapped[str] = mapped_column(String(100))
    umur: Mapped[int]
    status: Mapped[str] = mapped_column(Enum(*STATUS, name="status"))
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    tangan: Mapped[list["Tangan"]] = relationship(
        back_populates="orang", cascade="all, delete-orphan"
    )


class Tangan(Base):
    __tablename__ = "tangan"

    id: Mapped[int] = mapped_column(primary_key=True)
    orang_id: Mapped[int] = mapped_column(ForeignKey("orang.id", ondelete="CASCADE"))
    sisi: Mapped[str] = mapped_column(Enum(*SISI, name="sisi"))
    # Model yang membuat embedding. Embedding dari model berbeda tidak bisa dibandingkan.
    model: Mapped[str] = mapped_column(String(50))
    # Daftar embedding, satu per foto sampel, contoh: [[0.1, -0.2, ...], [0.1, -0.3, ...], ...]
    embedding: Mapped[list] = mapped_column(JSON)
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    orang: Mapped[Orang] = relationship(back_populates="tangan")


def buat_engine(url):
    """Hubungkan ke database. Database MySQL dan tabel-tabelnya dibuat otomatis kalau belum ada."""
    u = make_url(url)
    try:
        if u.get_backend_name() == "mysql":
            tanpa_nama_db = u._replace(database=None)
            with create_engine(tanpa_nama_db).connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{u.database}`"))
        engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(engine)
    except OperationalError as e:
        raise DatabaseTidakTersedia(
            f"MySQL tidak bisa dihubungi di {u.host}:{u.port or 3306}. "
            "Nyalakan MySQL di Laragon lalu jalankan ulang."
        ) from e
    return engine


def tambah_orang(s, nama, umur, status):
    orang = Orang(nama=nama, umur=umur, status=status)
    s.add(orang)
    s.flush()  # supaya orang.id langsung terisi
    return orang


def cari_tangan(s, orang_id, sisi):
    return s.scalar(select(Tangan).where(Tangan.orang_id == orang_id, Tangan.sisi == sisi))


def simpan_tangan(s, orang_id, sisi, embedding, model):
    """Simpan embedding satu tangan. Kalau tangan itu sudah ada, datanya diganti."""
    tangan = cari_tangan(s, orang_id, sisi)
    if tangan is None:
        tangan = Tangan(orang_id=orang_id, sisi=sisi)
        s.add(tangan)
    tangan.embedding = embedding
    tangan.model = model


def semua_tangan(s, model):
    """Semua tangan terdaftar yang dibuat dengan `model`, siap untuk pencocokan."""
    baris = s.scalars(select(Tangan).where(Tangan.model == model))
    return [
        {
            "tangan_id": t.id,
            "orang_id": t.orang.id,
            "nama": t.orang.nama,
            "sisi": t.sisi,
            "embedding": t.embedding,
        }
        for t in baris
    ]


def orang_ke_dict(orang):
    return {
        "id": orang.id,
        "nama": orang.nama,
        "umur": orang.umur,
        "status": orang.status,
        "tangan": [{"sisi": t.sisi, "model": t.model} for t in orang.tangan],
        "dibuat_pada": orang.dibuat_pada.isoformat(),
    }


def daftar_orang(s):
    return [orang_ke_dict(o) for o in s.scalars(select(Orang).order_by(Orang.nama))]


def ambil_orang(s, orang_id):
    orang = s.get(Orang, orang_id)
    return None if orang is None else orang_ke_dict(orang)


def hapus_orang(s, orang_id):
    """Hapus satu orang beserta tangannya. Kembalikan False kalau orangnya tidak ada."""
    orang = s.get(Orang, orang_id)
    if orang is None:
        return False
    s.delete(orang)
    return True


def hapus_semua(s):
    """Hapus semua orang beserta tangannya. Kembalikan jumlah orang yang dihapus."""
    semua = s.scalars(select(Orang)).all()
    for orang in semua:
        s.delete(orang)
    return len(semua)
