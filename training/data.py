"""data.py - turn the Tongji contactless palmprint photos into 128x128 palm ROIs and split them by palm.

Tongji layout: session1/ and session2/, the same file name in both sessions is the same palm, and every
10 numbers are one palm (00001-00010 = palm 0, 00011-00020 = palm 1, ...).

The ROI is cut with palm.ambil_roi, the same function the web app uses, so the model is trained on
exactly the kind of crop it will see from the webcam.

Run from the repo root:

    python -m training.data --asli data/tongji/asli --keluar data/roi
    python -m training.data --bawaan data/tongji/roi --keluar data/roi     # use Tongji's own ROI images instead
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

import training  # noqa: F401  (puts backend/ on sys.path)
import palm

SESI = ("session1", "session2")
EKSTENSI = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FOTO_PER_TELAPAK = 10
JUMLAH_DAFTAR = 8  # photos of session1 used to enrol a palm in the evaluation, like the 8 samples in the app


def id_telapak(nama_file):
    """Palm number from a Tongji file name: "00001.bmp" -> 0, "00011" -> 1."""
    return (int(Path(nama_file).stem) - 1) // FOTO_PER_TELAPAK


def _foto(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EKSTENSI) if Path(folder).is_dir() else []


def _baca(path):
    """Read any image as BGR (grey photos are converted), or None if it cannot be read."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3] if img.shape[2] == 4 else img


def siapkan_roi(folder_asli, folder_keluar, deteksi=palm.deteksi_tangan):
    """Cut the ROI of every photo with MediaPipe + palm.ambil_roi. Photos that fail are skipped and counted."""
    hasil = {"total": 0, "berhasil": 0, "gagal": []}
    for sesi in SESI:
        tujuan = Path(folder_keluar) / sesi
        for path in _foto(Path(folder_asli) / sesi):
            hasil["total"] += 1
            frame = _baca(path)
            deteksi_hasil = None if frame is None else deteksi(frame)
            roi = None if deteksi_hasil is None else palm.ambil_roi(frame, deteksi_hasil[0])
            if roi is None:
                hasil["gagal"].append(f"{sesi}/{path.name}")
                continue
            tujuan.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tujuan / f"{path.stem}.png"), roi)
            hasil["berhasil"] += 1
    return hasil


def salin_roi_bawaan(folder_roi, folder_keluar):
    """Use the ROI images that come with Tongji: grey, resized to 128x128, same layout as siapkan_roi."""
    hasil = {"total": 0, "berhasil": 0, "gagal": []}
    for sesi in SESI:
        tujuan = Path(folder_keluar) / sesi
        for path in _foto(Path(folder_roi) / sesi):
            hasil["total"] += 1
            img = _baca(path)
            if img is None:
                hasil["gagal"].append(f"{sesi}/{path.name}")
                continue
            tujuan.mkdir(parents=True, exist_ok=True)
            roi = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (palm.UKURAN, palm.UKURAN), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tujuan / f"{path.stem}.png"), roi)
            hasil["berhasil"] += 1
    return hasil


def roi_per_telapak(folder_roi, sesi):
    """{palm id: [ROI paths sorted by name]} of one session."""
    kelompok = {}
    for path in _foto(Path(folder_roi) / sesi):
        kelompok.setdefault(id_telapak(path.name), []).append(path)
    return kelompok


def telapak_lengkap(folder_roi):
    """Palms with enough ROIs to enrol (session1) and at least one scan (session2)."""
    s1, s2 = roi_per_telapak(folder_roi, "session1"), roi_per_telapak(folder_roi, "session2")
    return sorted(t for t, daftar in s1.items() if len(daftar) >= JUMLAH_DAFTAR and s2.get(t))


def buat_pembagian(folder_roi, seed=42):
    """Split the palms 70/10/20 into training, validation and test. A palm is in exactly one group.

    Splitting by palm (not by photo) means the test palms are never seen during training, like a
    new person who registers in the app.
    """
    ids = telapak_lengkap(folder_roi)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    jumlah_uji, jumlah_validasi = round(0.2 * n), round(0.1 * n)
    uji, validasi, latih = ids[:jumlah_uji], ids[jumlah_uji:jumlah_uji + jumlah_validasi], ids[jumlah_uji + jumlah_validasi:]
    return {"latih": sorted(latih), "validasi": sorted(validasi), "uji": sorted(uji)}


def _laporan(hasil, sumber, pembagian):
    persen = 100 * hasil["berhasil"] / hasil["total"] if hasil["total"] else 0.0
    baris = [f"Sumber ROI: {sumber}",
             f"Foto: {hasil['total']}, berhasil: {hasil['berhasil']} ({persen:.1f}%), gagal: {len(hasil['gagal'])}",
             f"Telapak: latih {len(pembagian['latih'])}, validasi {len(pembagian['validasi'])}, uji {len(pembagian['uji'])}"]
    if hasil["gagal"]:
        baris.append("Contoh foto gagal: " + ", ".join(hasil["gagal"][:20]))
    return "\n".join(baris), persen


def main(argv=None):
    ap = argparse.ArgumentParser(description="Siapkan ROI 128x128 dari dataset Tongji dan bagi per telapak.")
    sumber = ap.add_mutually_exclusive_group(required=True)
    sumber.add_argument("--asli", help="folder Original Images (berisi session1/ dan session2/)")
    sumber.add_argument("--bawaan", help="folder ROI Images bawaan Tongji (berisi session1/ dan session2/)")
    ap.add_argument("--keluar", default="data/roi", help="folder ROI hasil (default data/roi)")
    args = ap.parse_args(argv)

    if args.asli:
        hasil, nama = siapkan_roi(args.asli, args.keluar), "MediaPipe + palm.ambil_roi (sama dengan aplikasi)"
    else:
        hasil, nama = salin_roi_bawaan(args.bawaan, args.keluar), "ROI bawaan Tongji"
    if hasil["total"] == 0:
        print("Galat: tidak ada foto. Folder harus berisi session1/ dan session2/.", file=sys.stderr)
        return 1
    pembagian = buat_pembagian(args.keluar)
    Path("data").mkdir(exist_ok=True)
    Path("data/pembagian.json").write_text(json.dumps(pembagian, indent=1), encoding="utf-8")
    teks, persen = _laporan(hasil, nama, pembagian)
    Path("hasil").mkdir(exist_ok=True)
    Path("hasil/siapkan_data.txt").write_text(teks + "\n", encoding="utf-8")
    print(teks)
    if args.asli and persen < 80:
        print("\nKurang dari 80% foto berhasil dipotong. Pertimbangkan --bawaan data/tongji/roi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
