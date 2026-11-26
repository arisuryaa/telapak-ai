"""steps.py - menyusun kartu langkah yang ditampilkan di halaman Hasil."""

import base64

import cv2
import mediapipe as mp

HIJAU = (0, 200, 0)  # warna BGR
MERAH = (0, 0, 255)
KUNING = (0, 190, 255)


def ke_data_url(gambar, format=".jpg"):
    """Ubah gambar menjadi teks data URL supaya bisa langsung ditampilkan di browser."""
    ok, isi = cv2.imencode(format, gambar)
    if not ok:
        raise ValueError("gambar tidak bisa diubah")
    jenis = "image/jpeg" if format == ".jpg" else "image/png"
    return f"data:{jenis};base64," + base64.b64encode(isi.tobytes()).decode("ascii")


def gambar_titik_tangan(foto, titik):
    """Gambar kerangka tangan dan 21 titiknya di salinan foto."""
    tinggi, lebar = foto.shape[:2]
    hasil = foto.copy()
    piksel = [(int(p.x * lebar), int(p.y * tinggi)) for p in titik]
    for a, b in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(hasil, piksel[a], piksel[b], HIJAU, 2)
    for p in piksel:
        cv2.circle(hasil, p, 4, MERAH, -1)
    return hasil


def gambar_kotak_roi(roi):
    """Gambar kotak ROI dan garis pangkal jari di salinan foto yang sudah diluruskan."""
    hasil = roi["foto_lurus"].copy()
    x1, y1, x2, y2 = roi["kotak"]
    cv2.rectangle(hasil, (x1, y1), (x2, y2), HIJAU, 3)
    cv2.line(hasil, roi["garis"][0], roi["garis"][1], KUNING, 3)
    return hasil


def susun_langkah(hasil, jarak=None, batas=None):
    """Kembalikan daftar langkah untuk halaman Hasil.

    Langkah hanya muncul kalau datanya ada, jadi foto yang ditolak hanya menampilkan
    langkah yang sempat dilewatinya.
    """
    foto = hasil["foto"]
    langkah = [
        {
            "judul": "Foto dari kamera",
            "penjelasan": "Satu foto diambil dari webcam ketika tangan sudah memenuhi syarat.",
            "gambar": [{"label": "Foto asli", "src": ke_data_url(foto)}],
            "angka": {"resolusi": f"{foto.shape[1]}×{foto.shape[0]}"},
        }
    ]

    if hasil["titik"] is not None:
        langkah.append(
            {
                "judul": "Deteksi tangan",
                "penjelasan": "MediaPipe menemukan 21 titik pada tangan. Garis hijau adalah kerangka tangan.",
                "gambar": [
                    {"label": "21 titik", "src": ke_data_url(gambar_titik_tangan(foto, hasil["titik"]))}
                ],
                "angka": {"jumlah titik": 21, "sisi": hasil["sisi"]},
            }
        )

    roi = hasil["roi"]
    if roi is not None:
        langkah.append(
            {
                "judul": "Telapak diluruskan",
                "penjelasan": (
                    "Foto diputar sampai garis pangkal jari (kuning) mendatar, "
                    "lalu kotak telapak (hijau) dipotong."
                ),
                "gambar": [{"label": "Foto diluruskan", "src": ke_data_url(gambar_kotak_roi(roi))}],
                "angka": {"sudut putar": f"{roi['sudut']:.1f}°"},
            }
        )
        langkah.append(
            {
                "judul": "ROI abu-abu",
                "penjelasan": "Kotak telapak diubah menjadi abu-abu lalu diperkecil ke 128×128.",
                "gambar": [
                    {"label": "Potongan berwarna", "src": ke_data_url(roi["potongan"], ".png")},
                    {"label": "Abu-abu 128×128", "src": ke_data_url(roi["roi"], ".png")},
                ],
                "angka": {"ukuran ROI": "128×128"},
            }
        )

    if hasil["embedding"] is not None:
        langkah.append(
            {
                "judul": "Model: embedding",
                "penjelasan": "Model mengubah ROI menjadi daftar angka (embedding).",
                "gambar": [],
                "angka": {"jumlah angka": len(hasil["embedding"])},
            }
        )

    if jarak is not None:
        langkah.append(
            {
                "judul": "Pencocokan",
                "penjelasan": (
                    "Embedding scan dibandingkan dengan setiap tangan terdaftar memakai jarak kosinus "
                    "(0 = sama persis). Jarak terkecil di bawah batas berarti dikenali."
                ),
                "gambar": [],
                "angka": {"batas": batas},
                "jarak": jarak,
            }
        )

    for nomor, l in enumerate(langkah, start=1):
        l["no"] = nomor
    return langkah
