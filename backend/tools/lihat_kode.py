"""lihat_kode.py - look at the numbers a palm turns into.

Run from the backend folder. Three ways in:

    python tools/lihat_kode.py --daftar                      # list the hands saved in the database
    python tools/lihat_kode.py --id 5 --sampel 1             # print the numbers of sample 1 of hand 5 (from MySQL)
    python tools/lihat_kode.py --foto data_uji/dewa_kanan/x.jpg   # run one photo through the app's pipeline and print its numbers
    python tools/lihat_kode.py --cocokkan data_uji/dewa_kanan/x.jpg   # match one photo against the database, sample by sample

Add --gambar folder to also save a picture (colour map) of the code. With --id it saves all 8 samples.

What you see is the stored pattern: a 128x128 table of small numbers.
  Canny: . = no edge, 1..6 = the direction of the edge line at that pixel
  Gabor: 0..5 = the direction of the strongest filter at that pixel
The numbers are only a small window of the table (default 20 rows x 40 columns); move it with --y and --x.
The database holds no photos, so --id shows exactly what is compared during a scan.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import palm` works from anywhere

import metode as metode_mod  # noqa: E402
import palm  # noqa: E402
from app import DATABASE_DEFAULT  # noqa: E402
from db import Orang, Telapak, bytes_ke_kode, muat_galeri  # noqa: E402
from pipeline import analisis  # noqa: E402

WARNA = [(46, 74, 217), (0, 169, 242), (80, 145, 47), (201, 167, 32), (224, 99, 53), (159, 68, 184)]  # BGR, 6 directions
# Gabor: number = the filter angle (0 = vertical lines, 3 = horizontal lines).
ARTI_GABOR = ["tegak", "miring /, curam", "miring /, landai", "datar", "miring \\, landai", "miring \\, curam"]
# Canny: number 1..6 = the angle of the line from horizontal, turning clockwise on the picture, 30 degrees per number.
ARTI_CANNY = ["0-30 derajat: hampir datar, turun ke kanan", "30-60 derajat: miring \\", "60-90 derajat: hampir tegak, turun ke kanan",
              "90-120 derajat: hampir tegak, turun ke kiri", "120-150 derajat: miring /", "150-180 derajat: hampir datar, turun ke kiri"]


def potongan_angka(kode, metode, y=44, x=44, tinggi=20, lebar=40):
    """A window of the code as text. For Canny a 0 (no edge) is shown as a dot so the lines stand out."""
    H, W = kode.shape
    y, x = max(0, min(y, H - tinggi)), max(0, min(x, W - lebar))
    jendela = kode[y:y + tinggi, x:x + lebar]
    kepala = "      " + "".join(str((x + k) % 10) for k in range(lebar))
    baris = [f"x={x}..{x + lebar - 1}", kepala]
    for i, row in enumerate(jendela):
        sel = "".join("." if (metode == "canny" and v == 0) else str(int(v)) for v in row)
        baris.append(f"y={y + i:<3} {sel}")
    return "\n".join(baris)


def ringkasan_kode(kode, metode):
    """How many pixels carry each number, in words."""
    jumlah = np.bincount(kode.ravel(), minlength=7)
    if metode == "canny":
        tepi = int((kode > 0).sum())
        baris = [f"piksel: {kode.size}, tepi: {tepi} ({100 * tepi / kode.size:.1f}%), bukan tepi: {kode.size - tepi}"]
        baris += [f"  angka {d + 1} = garis {ARTI_CANNY[d]:<44} {int(jumlah[d + 1]):>5} piksel" for d in range(6)]
    else:
        baris = [f"piksel: {kode.size}"]
        baris += [f"  angka {d} = garis {ARTI_GABOR[d]:<17} {int(jumlah[d]):>5} piksel" for d in range(6)]
    return "\n".join(baris)


def kode_ke_gambar(kode, metode, besar=4):
    """The code as a colour picture: one colour per direction (Canny: edges on black)."""
    warna = np.array(WARNA, np.uint8)
    gambar = np.zeros(kode.shape + (3,), np.uint8)
    if metode == "canny":
        gambar[kode > 0] = warna[kode[kode > 0] - 1]
    else:
        gambar = warna[kode]
    return cv2.resize(gambar, None, fx=besar, fy=besar, interpolation=cv2.INTER_NEAREST)


def lembar_sampel(semua, metode):
    """All samples of a hand side by side (two rows), each labelled."""
    ubin = []
    for i, kode in enumerate(semua, 1):
        g = cv2.copyMakeBorder(kode_ke_gambar(kode, metode, 2), 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(g, f"sampel {i}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        ubin.append(g)
    sebaris = (len(ubin) + 1) // 2
    atas, bawah = ubin[:sebaris], ubin[sebaris:]
    bawah += [np.full_like(ubin[0], 255)] * (sebaris - len(bawah))
    return np.vstack([np.hstack(atas), np.hstack(bawah)])


def kode_dari_foto(path, metode, roi_skala, deteksi=palm.deteksi_tangan):
    """Run one photo through the same analysis the server runs for a sample. Raises ValueError if it is rejected."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Foto tidak bisa dibaca: {path}")
    a = analisis(frame, deteksi=deteksi, roi_skala=roi_skala, metode=metode)
    if a.gagal:
        raise ValueError(f"Foto ditolak seperti di aplikasi: {a.gagal}")
    return a.kode


def buka_engine():
    from sqlalchemy import create_engine
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return create_engine(os.environ.get("DATABASE_URL", DATABASE_DEFAULT))


def daftar_tangan(engine):
    """[(telapak id, name, side, method, ROI size, bytes)] of every saved hand."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    with Session(engine) as s:
        baris = s.execute(select(Telapak, Orang).join(Telapak.orang).order_by(Telapak.id)).all()
        return [(t.id, o.nama, t.sisi, t.metode, t.roi_skala, len(t.kode)) for t, o in baris]


def kode_dari_db(engine, telapak_id):
    """(info, codes of shape (8, 128, 128)) of one saved hand. Raises ValueError if the id does not exist."""
    from sqlalchemy.orm import Session
    with Session(engine) as s:
        t = s.get(Telapak, telapak_id)
        if t is None:
            raise ValueError(f"Tidak ada tangan dengan id {telapak_id}. Lihat daftar dengan --daftar.")
        info = {"id": t.id, "nama": t.orang.nama, "sisi": t.sisi, "metode": t.metode, "roi_skala": t.roi_skala,
                "ukuran": len(t.kode), "dibuat_pada": t.dibuat_pada}
        return info, bytes_ke_kode(t.kode)


def cocokkan(engine, kode, metode, roi_skala):
    """Compare one code with every saved hand the app would compare it with, as the scan does.

    Returns [{"telapak_id", "nama", "sisi", "jarak", "per_sampel"}], closest hand first. A hand's distance is
    the distance to its closest sample. Hands saved with another method or box size are left out.
    """
    from sqlalchemy.orm import Session
    with Session(engine) as s:
        galeri = muat_galeri(s)
    bentuk = palm.versi_bentuk(roi_skala)
    hasil = []
    for g in galeri:
        if g["metode"] != metode or abs(g["roi_skala"] - roi_skala) > 1e-9 or g["roi_bentuk"] != bentuk:
            continue
        per_sampel = [float(d) for d in metode_mod.jarak_per_sampel(metode, kode, g["kode"])]
        hasil.append({"telapak_id": g["telapak_id"], "nama": g["nama"], "sisi": g["sisi"],
                      "jarak": min(per_sampel), "per_sampel": per_sampel})
    return sorted(hasil, key=lambda h: h["jarak"])


def cetak_cocokkan(hasil, batas, jumlah_semua):
    if not hasil:
        print(f"Tidak ada tangan yang bisa dibandingkan ({jumlah_semua} tangan di database, "
              "tetapi metode atau ukuran kotaknya berbeda)." if jumlah_semua else "Database masih kosong.")
        return
    print(f"Dibandingkan dengan {len(hasil)} tangan. Jarak 0 = sama persis; di bawah batas {batas:g} = dikenali.\n")
    for n, h in enumerate(hasil):
        tanda = ""
        if n == 0:
            tanda = "  <- terdekat, DIKENALI" if h["jarak"] < batas else "  <- terdekat, tetapi di atas batas: TIDAK DIKENALI"
        print(f"{h['nama']} ({h['sisi']}, id {h['telapak_id']})  jarak {h['jarak']:.3f}{tanda}")
        terbaik = h["per_sampel"].index(h["jarak"])
        for i, d in enumerate(h["per_sampel"]):
            print(f"  sampel {i + 1}  {d:.3f}" + ("  <- paling mirip, ini jarak tangan ini" if i == terbaik else ""))
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lihat angka kode telapak (dari database atau dari satu foto).")
    sumber = ap.add_mutually_exclusive_group(required=True)
    sumber.add_argument("--daftar", action="store_true", help="tampilkan daftar tangan yang tersimpan di database")
    sumber.add_argument("--id", type=int, help="id tangan di tabel telapak (lihat --daftar)")
    sumber.add_argument("--foto", help="jalankan satu foto lewat pipeline aplikasi dan tampilkan angkanya")
    sumber.add_argument("--cocokkan", metavar="FOTO", help="cocokkan satu foto dengan semua tangan di database")
    ap.add_argument("--batas", type=float, default=None, help="batas jarak untuk --cocokkan (default: batas bawaan metode)")
    ap.add_argument("--sampel", type=int, default=1, help="nomor sampel 1-8 untuk --id (default 1)")
    ap.add_argument("--metode", default=None, help="canny atau gabor untuk --foto (default: pengaturan METODE)")
    ap.add_argument("--skala", type=float, default=None, help="ukuran kotak ROI untuk --foto (default: ROI_SKALA)")
    ap.add_argument("--y", type=int, default=44, help="baris awal jendela angka")
    ap.add_argument("--x", type=int, default=44, help="kolom awal jendela angka")
    ap.add_argument("--tinggi", type=int, default=20, help="tinggi jendela angka")
    ap.add_argument("--lebar", type=int, default=40, help="lebar jendela angka")
    ap.add_argument("--gambar", help="simpan gambar warna kode ke folder ini")
    args = ap.parse_args(argv)

    from sqlalchemy.exc import OperationalError
    try:
        return _jalankan(args)
    except OperationalError as err:
        print(f"Galat: MySQL tidak bisa dihubungi. Nyalakan MySQL di Laragon lalu coba lagi. Detail: {err.orig}",
              file=sys.stderr)
        return 1


def _metode_dan_skala(args):
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    metode = metode_mod.periksa(args.metode or os.environ.get("METODE", metode_mod.METODE_DEFAULT))
    skala = args.skala if args.skala is not None else float(os.environ.get("ROI_SKALA", "1.0"))
    return metode, skala


def _jalankan(args):
    try:
        if args.cocokkan:
            metode, skala = _metode_dan_skala(args)
            batas = args.batas if args.batas is not None else metode_mod.pengaturan(metode)["batas_default"]
            kode = kode_dari_foto(args.cocokkan, metode, skala, deteksi=palm.deteksi_tangan)
            engine = buka_engine()
            print(f"Foto {args.cocokkan}, metode {metode}, kotak {skala:g}x, batas {batas:g}.")
            print(f"Pola foto ini: {ringkasan_kode(kode, metode).splitlines()[0]}")
            cetak_cocokkan(cocokkan(engine, kode, metode, skala), batas, len(daftar_tangan(engine)))
            return 0
        if args.daftar or args.id is not None:
            engine = buka_engine()
        if args.daftar:
            tangan = daftar_tangan(engine)
            print("id   nama                  sisi   metode  ukuran kotak  byte")
            for i, nama, sisi, metode, skala, ukuran in tangan:
                print(f"{i:<4} {nama:<21} {sisi:<6} {metode:<7} {skala:<13g} {ukuran}")
            if not tangan:
                print("(belum ada tangan terdaftar)")
            return 0
        if args.id is not None:
            info, semua = kode_dari_db(engine, args.id)
            metode, label = info["metode"], f"sampel {args.sampel} dari {len(semua)}"
            if not 1 <= args.sampel <= len(semua):
                raise ValueError(f"--sampel harus antara 1 dan {len(semua)}")
            kode = semua[args.sampel - 1]
            print(f"Tangan {info['sisi']} milik {info['nama']} (id {info['id']}), metode {metode}, kotak {info['roi_skala']:g}x, "
                  f"{info['ukuran']} byte = {len(semua)} sampel x 128 x 128, dibuat {info['dibuat_pada']}")
        else:
            metode, skala = _metode_dan_skala(args)
            kode, semua, label = kode_dari_foto(args.foto, metode, skala, deteksi=palm.deteksi_tangan), None, "hasil dari foto ini"
            print(f"Foto {args.foto}, metode {metode}, kotak {skala:g}x. Ini kode yang akan disimpan sebagai satu sampel.")
    except ValueError as err:
        print(f"Galat: {err}", file=sys.stderr)
        return 1

    print(f"\n{label}\n{ringkasan_kode(kode, metode)}\n")
    print(potongan_angka(kode, metode, args.y, args.x, args.tinggi, args.lebar))
    if args.gambar:
        tujuan = Path(args.gambar)
        tujuan.mkdir(parents=True, exist_ok=True)
        nama_file = tujuan / (f"tangan_{args.id}.png" if semua is not None else "kode_foto.png")
        cv2.imwrite(str(nama_file), lembar_sampel(semua, metode) if semua is not None else kode_ke_gambar(kode, metode))
        print(f"\nGambar disimpan di {nama_file}. Isinya pola telapak Anda (data biometrik), jangan dibagikan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
