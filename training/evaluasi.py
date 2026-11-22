"""evaluasi.py - measure the trained model on palms it has never seen.

For every palm: the first 8 ROIs of session 1 are its enrolment (like the 8 samples in the app) and every
ROI of session 2 (taken about two months later) is a scan. The threshold is picked on the validation palms
and stored in the model file; the numbers for the report are measured on the test palms.

Run from the repo root:

    python -m training.evaluasi --model model/telapak.pt --roi data/roi
    python -m training.evaluasi --model model/telapak.pt --roi data/roi --sendiri data/sendiri
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import training  # noqa: F401  (puts backend/ on sys.path)
from training.data import JUMLAH_DAFTAR, roi_per_telapak
from training.metrik import eer, roc, skor_identifikasi, tingkat_salah

POTONGAN = 256  # ROIs per batch through the model


def _embedding(pengenal, paths):
    roi = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    return np.concatenate([pengenal.embedding_banyak(roi[i:i + POTONGAN]) for i in range(0, len(roi), POTONGAN)])


def evaluasi(pengenal, folder_roi, ids, jumlah_daftar=JUMLAH_DAFTAR):
    """Scores of `pengenal` on the palms `ids`. Palms without enough enrolment or scan photos are skipped."""
    s1, s2 = roi_per_telapak(folder_roi, "session1"), roi_per_telapak(folder_roi, "session2")
    pakai = [t for t in ids if len(s1.get(t, [])) >= jumlah_daftar and s2.get(t)]
    if len(pakai) < 2:
        raise ValueError(f"Evaluasi butuh minimal 2 telapak lengkap, hanya ada {len(pakai)}")
    daftar = {t: _embedding(pengenal, s1[t][:jumlah_daftar]) for t in pakai}
    scan = [(t, e) for t in pakai for e in _embedding(pengenal, s2[t])]
    asli, palsu, rank1 = skor_identifikasi(daftar, scan)
    nilai_eer, batas = eer(asli, palsu)
    return {"eer": nilai_eer, "batas": batas, "rank1": rank1, "jumlah_telapak": len(pakai),
            "jumlah_scan": len(scan), "asli": asli, "palsu": palsu}


def ringkas(hasil, batas):
    """The numbers of one evaluation at a given threshold, ready for JSON."""
    far, frr = tingkat_salah(hasil["asli"], hasil["palsu"], batas)
    return {"eer": round(hasil["eer"], 4), "rank1": round(hasil["rank1"], 4), "batas": round(float(batas), 4),
            "far_pada_batas": round(float(far[0]), 4), "frr_pada_batas": round(float(frr[0]), 4),
            "jumlah_telapak": hasil["jumlah_telapak"], "jumlah_scan": hasil["jumlah_scan"],
            "jumlah_pasangan_asli": int(len(hasil["asli"])), "jumlah_pasangan_palsu": int(len(hasil["palsu"]))}


def gambar_roc(hasil, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    far, tar = roc(hasil["asli"], hasil["palsu"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(np.maximum(far, 1e-5), tar, color="#2563eb")
    ax.set_xscale("log")
    ax.set_xlabel("FAR (orang lain diterima)")
    ax.set_ylabel("TAR (pemilik dikenali)")
    ax.set_title(f"Kurva ROC (EER {100 * hasil['eer']:.2f}%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gambar_histogram(hasil, batas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(1.2, float(hasil["palsu"].max())), 60)
    ax.hist(hasil["palsu"], bins=bins, density=True, alpha=0.6, color="#ef4444", label="Beda telapak")
    ax.hist(hasil["asli"], bins=bins, density=True, alpha=0.6, color="#16a34a", label="Sama telapak")
    ax.axvline(batas, color="black", linestyle="--", label=f"Batas {batas:.3f}")
    ax.set_xlabel("Jarak (1 - kosinus)")
    ax.set_ylabel("Jumlah pasangan (dinormalisasi)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluasi_sendiri(pengenal, folder, batas, deteksi=None):
    """Photos of the owner: one folder per hand, the first 8 photos enrol it, the rest are scans."""
    import palm
    deteksi = deteksi or palm.deteksi_tangan
    daftar, scan, nama = {}, [], {}
    for t, sub in enumerate(sorted(p for p in Path(folder).iterdir() if p.is_dir())):
        roi = []
        for path in sorted(sub.iterdir()):
            frame = cv2.imread(str(path))
            hasil = None if frame is None else deteksi(frame)
            r = None if hasil is None else palm.ambil_roi(frame, hasil[0])
            if r is not None:
                roi.append(r)
        if len(roi) <= JUMLAH_DAFTAR:
            continue
        nama[t] = sub.name
        e = pengenal.embedding_banyak(roi)
        daftar[t] = e[:JUMLAH_DAFTAR]
        scan += [(t, v) for v in e[JUMLAH_DAFTAR:]]
    if len(daftar) < 2:
        return None
    asli, palsu, rank1 = skor_identifikasi(daftar, scan)
    far, frr = tingkat_salah(asli, palsu, batas)
    return {"tangan": [nama[t] for t in sorted(daftar)], "jumlah_scan": len(scan), "rank1": round(rank1, 4),
            "jarak_asli_rata": round(float(asli.mean()), 4), "jarak_palsu_rata": round(float(palsu.mean()), 4),
            "far_pada_batas": round(float(far[0]), 4), "frr_pada_batas": round(float(frr[0]), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluasi model pada telapak yang belum pernah dilihat.")
    ap.add_argument("--model", default="model/telapak.pt")
    ap.add_argument("--roi", default="data/roi")
    ap.add_argument("--pembagian", default="data/pembagian.json")
    ap.add_argument("--sendiri", help="folder foto pemilik (satu folder per tangan)")
    args = ap.parse_args(argv)

    from model import Pengenal, simpan_model
    try:
        pengenal = Pengenal.dari_file(args.model)
        pembagian = json.loads(Path(args.pembagian).read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        print(f"Galat: {err}", file=sys.stderr)
        return 1

    validasi = evaluasi(pengenal, args.roi, pembagian["validasi"])
    batas = validasi["batas"]
    simpan_model(args.model, pengenal.jaringan, versi=pengenal.versi, batas=batas, info=pengenal.info)
    uji = evaluasi(pengenal, args.roi, pembagian["uji"])

    Path("hasil").mkdir(exist_ok=True)
    laporan = {"model_versi": pengenal.versi, "validasi": ringkas(validasi, batas), "uji": ringkas(uji, batas)}
    gambar_roc(uji, "hasil/roc.png")
    gambar_histogram(uji, batas, "hasil/histogram_jarak.png")
    if args.sendiri:
        laporan["foto_sendiri"] = evaluasi_sendiri(pengenal, args.sendiri, batas)
    Path("hasil/evaluasi.json").write_text(json.dumps(laporan, indent=2), encoding="utf-8")

    u = laporan["uji"]
    print(f"Model {pengenal.versi}. Batas dari data validasi: {batas:.4f} (disimpan ke {args.model})")
    print(f"Data uji: {u['jumlah_telapak']} telapak, {u['jumlah_scan']} scan")
    print(f"  EER     {100 * u['eer']:.2f}%")
    print(f"  Rank-1  {100 * u['rank1']:.2f}%   (scan yang telapak terdekatnya benar)")
    print(f"  Pada batas: FAR {100 * u['far_pada_batas']:.3f}%, FRR {100 * u['frr_pada_batas']:.2f}%")
    if laporan.get("foto_sendiri"):
        s = laporan["foto_sendiri"]
        print(f"Foto sendiri ({', '.join(s['tangan'])}): rank-1 {100 * s['rank1']:.1f}%, "
              f"FAR {100 * s['far_pada_batas']:.1f}%, FRR {100 * s['frr_pada_batas']:.1f}%")
    print("Grafik: hasil/roc.png, hasil/histogram_jarak.png; angka: hasil/evaluasi.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
