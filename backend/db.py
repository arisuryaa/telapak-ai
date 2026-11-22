"""db.py - MySQL models and the data functions the API uses.

Only palm embeddings are stored, never photos. Each person can have one set
per hand side (kiri / kanan), each set holding 8 embeddings of 128 float32 numbers.
"""
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import (DateTime, Enum, ForeignKey, Integer, LargeBinary, String,
                        UniqueConstraint, create_engine, delete, select, text)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

STATUS = ("mahasiswa", "dosen", "staf", "tamu")
SISI = ("kiri", "kanan")
DIMENSI = 128  # numbers per embedding


class DatabaseTidakTersedia(Exception):
    """The database server could not be reached."""


class TelapakSudahAda(Exception):
    """This person already has a code set for this hand side."""


class Base(DeclarativeBase):
    pass


def _sekarang():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Orang(Base):
    __tablename__ = "orang"

    id: Mapped[int] = mapped_column(primary_key=True)
    nama: Mapped[str] = mapped_column(String(100))
    umur: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Enum(*STATUS, name="status"))
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime, default=_sekarang)
    telapak: Mapped[list["Telapak"]] = relationship(back_populates="orang", cascade="all, delete-orphan")


class Telapak(Base):
    __tablename__ = "telapak"
    __table_args__ = (UniqueConstraint("orang_id", "sisi"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    orang_id: Mapped[int] = mapped_column(ForeignKey("orang.id", ondelete="CASCADE"))
    sisi: Mapped[str] = mapped_column(Enum(*SISI, name="sisi"))
    # 8 embeddings x 128 float32 = 4096 bytes
    kode: Mapped[bytes] = mapped_column(LargeBinary)
    # the model that made these embeddings; embeddings of two different trainings cannot be compared
    model_versi: Mapped[str] = mapped_column(String(20), default="")
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime, default=_sekarang)
    orang: Mapped[Orang] = relationship(back_populates="telapak")


def _pesan_tidak_tersedia(url, err):
    detail = getattr(err, "orig", None) or err
    return (f"MySQL tidak bisa dihubungi di {url.host}:{url.port or 3306}. "
            f"Nyalakan MySQL di Laragon lalu jalankan ulang. Detail: {detail}")


def _url_server(u):
    """Same server URL without a database, used to create the database first.
    URL.set(database=None) would keep the old value, so replace it directly."""
    return u._replace(database=None)


def buat_engine(url):
    """Connect, create the MySQL database if missing, and create the tables."""
    u = make_url(url)
    try:
        if u.get_backend_name() == "mysql":
            nama_db = u.database
            with create_engine(_url_server(u)).connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{nama_db}` CHARACTER SET utf8mb4"))
                conn.commit()
        engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(engine)
    except OperationalError as e:
        raise DatabaseTidakTersedia(_pesan_tidak_tersedia(u, e)) from e
    return engine


def kode_ke_bytes(kode):
    return np.asarray(kode, np.float32).tobytes()


def bytes_ke_kode(b):
    return np.frombuffer(b, np.float32).reshape(-1, DIMENSI).copy()


def tambah_orang(s, nama, umur, status):
    o = Orang(nama=nama, umur=umur, status=status)
    s.add(o)
    s.flush()
    return o


def simpan_telapak(s, orang_id, sisi, kode, ganti=False, model_versi=""):
    """Store the embeddings of one hand. Raises TelapakSudahAda unless `ganti` is True."""
    ada = s.scalar(select(Telapak).where(Telapak.orang_id == orang_id, Telapak.sisi == sisi))
    if ada is not None:
        if not ganti:
            raise TelapakSudahAda(sisi)
        ada.kode = kode_ke_bytes(kode)
        ada.model_versi = model_versi
        s.flush()
        return ada
    t = Telapak(orang_id=orang_id, sisi=sisi, kode=kode_ke_bytes(kode), model_versi=model_versi)
    s.add(t)
    s.flush()
    return t


def muat_galeri(s):
    """Every enrolled code set, ready for matching."""
    rows = s.execute(select(Telapak, Orang).join(Telapak.orang)).all()
    return [{
        "telapak_id": t.id, "orang_id": o.id, "nama": o.nama,
        "sisi": t.sisi, "kode": bytes_ke_kode(t.kode), "model_versi": t.model_versi,
    } for t, o in rows]


def orang_ke_dict(o):
    return {
        "id": o.id, "nama": o.nama, "umur": o.umur, "status": o.status,
        "tangan": sorted(t.sisi for t in o.telapak),
        "versi_tangan": {t.sisi: t.model_versi for t in o.telapak},
        "dibuat_pada": o.dibuat_pada.isoformat(),
    }


def daftar_orang(s):
    return [orang_ke_dict(o) for o in s.scalars(select(Orang).order_by(Orang.nama))]


def ambil_orang(s, orang_id):
    o = s.get(Orang, orang_id)
    return None if o is None else orang_ke_dict(o)


def hapus_orang(s, orang_id):
    o = s.get(Orang, orang_id)
    if o is None:
        return False
    s.delete(o)  # cascades to the person's code sets
    return True


def hapus_semua(s):
    jumlah = len(s.scalars(select(Orang)).all())
    s.execute(delete(Telapak))
    s.execute(delete(Orang))
    return jumlah
