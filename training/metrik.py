"""metrik.py - how good a palm recogniser is, from the distances it gives.

  asli  (genuine):  distance from a scan to the palm it really belongs to   (should be small)
  palsu (impostor): distance from a scan to every other palm               (should be large)

A scan is accepted when its distance is below the threshold (batas), as in the app.
  FRR = share of genuine scans rejected     (the owner is not recognised)
  FAR = share of impostor pairs accepted   (someone else is let in)
  EER = the error rate at the threshold where FAR and FRR are equal (smaller is better)
"""
import numpy as np


def jarak_kosinus(a, b):
    """Cosine distance between every row of a [N,128] and every row of b [M,128] (both unit length)."""
    return np.clip(1.0 - np.asarray(a, np.float32) @ np.asarray(b, np.float32).T, 0.0, 2.0)


def _kandidat_batas(asli, palsu):
    nilai = np.unique(np.concatenate([asli, palsu]))
    tengah = (nilai[:-1] + nilai[1:]) / 2
    return np.concatenate([[nilai[0] - 1e-6], tengah, [nilai[-1] + 1e-6]])


def tingkat_salah(asli, palsu, batas):
    """(FAR, FRR) for each threshold in `batas` (accept when distance < threshold)."""
    asli, palsu = np.sort(np.asarray(asli)), np.sort(np.asarray(palsu))
    batas = np.atleast_1d(batas)
    far = np.searchsorted(palsu, batas, side="left") / len(palsu)
    frr = 1.0 - np.searchsorted(asli, batas, side="left") / len(asli)
    return far, frr


def eer(asli, palsu):
    """(EER, threshold at the EER)."""
    batas = _kandidat_batas(asli, palsu)
    far, frr = tingkat_salah(asli, palsu, batas)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(batas[i])


def roc(asli, palsu):
    """(FAR, TAR) over every threshold, from accepting nothing to accepting everything."""
    far, frr = tingkat_salah(asli, palsu, _kandidat_batas(asli, palsu))
    return far, 1.0 - frr


def skor_identifikasi(daftar, scan):
    """Compare every scan with every enrolled palm, as the app does.

    daftar: {palm id: enrolled embeddings [k,128]}; scan: [(palm id, embedding [128])].
    The distance to a palm is the distance to its closest enrolled sample.
    Returns (genuine distances, impostor distances, rank-1 = share of scans whose closest palm is the right one).
    """
    ids = sorted(daftar)
    label = np.array([t for t, _ in scan])
    s = np.stack([e for _, e in scan])
    d = np.stack([jarak_kosinus(s, daftar[t]).min(axis=1) for t in ids], axis=1)  # [scans, palms]
    sendiri = label[:, None] == np.array(ids)[None, :]
    rank1 = float(np.mean(np.array(ids)[d.argmin(axis=1)] == label))
    return d[sendiri], d[~sendiri], rank1
