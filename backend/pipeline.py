"""pipeline.py - memproses satu foto dari kamera, langkah demi langkah.

Proses berhenti di pemeriksaan pertama yang gagal, dan alasannya disimpan di hasil["gagal"].
"""

import base64

import cv2
import numpy as np

import model
import palm
from validation import cek_kualitas_roi, jari_terbuka, menghadap_kamera, sisi_dari_label


def baca_foto(data_url):
    """Ubah foto dari browser (data URL base64 JPEG/PNG) menjadi gambar BGR. None kalau rusak."""
    try:
        data = base64.b64decode(data_url.split(",", 1)[-1], validate=True)
    except (ValueError, TypeError, AttributeError):
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def proses_foto(foto):
    """Deteksi tangan, cek aturan, potong ROI, lalu buat embedding dengan model.

    Kembalikan dict. Kunci yang belum tercapai bernilai None.
    """
    hasil = {"foto": foto, "titik": None, "sisi": None, "roi": None, "embedding": None, "gagal": None}
    tinggi, lebar = foto.shape[:2]

    # 1. Cari tangan
    deteksi = palm.deteksi_tangan(foto)
    if deteksi is None:
        hasil["gagal"] = "Tangan tidak terdeteksi"
        return hasil
    titik, label = deteksi
    hasil["titik"] = titik
    hasil["sisi"] = sisi_dari_label(label)

    # 2. Cek posisi tangan
    if not jari_terbuka(titik, lebar, tinggi):
        hasil["gagal"] = "Buka semua jari"
        return hasil
    if not menghadap_kamera(titik, hasil["sisi"], lebar, tinggi):
        hasil["gagal"] = "Tunjukkan telapak tangan, bukan punggung tangan"
        return hasil

    # 3. Potong ROI dan cek kualitasnya
    roi = palm.potong_roi(foto, titik)
    if roi is None:
        hasil["gagal"] = "Tangan terlalu dekat/jauh atau keluar dari bingkai"
        return hasil
    hasil["roi"] = roi
    alasan = cek_kualitas_roi(roi["roi"])
    if alasan:
        hasil["gagal"] = alasan
        return hasil

    # 4. Ubah ROI menjadi embedding dengan model
    if not model.model_siap():
        hasil["gagal"] = "Model belum dibuat. Isi fungsi buat_embedding di backend/model.py"
        return hasil
    hasil["embedding"] = [float(x) for x in model.buat_embedding(roi["roi"])]
    return hasil
