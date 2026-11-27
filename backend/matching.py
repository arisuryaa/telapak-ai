"""matching.py - membandingkan embedding hasil scan dengan embedding yang tersimpan."""

import numpy as np


def jarak_kosinus(a, b):
    """Jarak antara dua embedding: 0 = arahnya sama persis, makin besar makin berbeda (maks. 2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    kosinus = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(1 - kosinus)


def hitung_jarak(embedding_scan, daftar_tangan):
    """Hitung jarak scan ke setiap tangan terdaftar, urut dari yang paling mirip.

    Setiap tangan punya beberapa sampel. Jarak tangan = jarak ke sampelnya yang paling mirip.
    """
    hasil = []
    for t in daftar_tangan:
        jarak_sampel = [jarak_kosinus(embedding_scan, sampel) for sampel in t["embedding"]]
        hasil.append(
            {
                "tangan_id": t["tangan_id"],
                "orang_id": t["orang_id"],
                "nama": t["nama"],
                "sisi": t["sisi"],
                "jarak": min(jarak_sampel),
            }
        )
    hasil.sort(key=lambda d: d["jarak"])
    return hasil
