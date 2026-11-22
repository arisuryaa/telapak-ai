"""steps.py - turn one analysis into the step cards shown on the results page."""
import base64

import cv2
import mediapipe as mp
import numpy as np


# The embedding picture: negative numbers blue, zero white, positive red (hex, shown in the legend)
WARNA_NEGATIF, WARNA_NOL, WARNA_POSITIF = "#2563eb", "#ffffff", "#dc2626"


def ke_data_url(img, fmt=".png"):
    ok, buf = cv2.imencode(fmt, img)
    if not ok:
        raise ValueError("image could not be encoded")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _hex_ke_bgr(hex_warna):
    r, g, b = (int(hex_warna[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


def gambar_embedding(e, besar=16):
    """The 128 numbers as an 8x16 grid of coloured squares: blue below zero, red above, white near zero."""
    v = np.asarray(e, np.float32).reshape(8, 16)
    s = (v / (float(np.abs(v).max()) or 1.0))[..., None]  # -1..1
    nol, neg, pos = (np.array(_hex_ke_bgr(c), np.float32) for c in (WARNA_NOL, WARNA_NEGATIF, WARNA_POSITIF))
    img = np.where(s < 0, nol + (neg - nol) * -s, nol + (pos - nol) * s).astype(np.uint8)
    img = cv2.resize(img, None, fx=besar, fy=besar, interpolation=cv2.INTER_NEAREST)
    img[::besar, :] = img[:, ::besar] = 200  # thin grey grid so each number is a visible square
    return img


def _gambar_landmark(frame, lm):
    """Draw the hand skeleton and the 21 points on a copy of the frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    titik = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(out, titik[a], titik[b], (0, 200, 0), 2)
    for p in titik:
        cv2.circle(out, p, 4, (0, 0, 255), -1)
    return out


_WARNA_GARIS = (0, 190, 255)  # BGR amber, the line between the two finger bases


def _gambar_sebelum_diluruskan(frame, detail):
    """The original photo with the ROI box and the finger-base line drawn where they really are.

    The box is cut from the rotated photo, so its corners are mapped back through the
    inverse of the rotation to show where that region sits on the photo before it was rotated.
    """
    x1, y1, x2, y2 = detail["kotak"]
    pojok_lurus = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    asli = cv2.transform(pojok_lurus, cv2.invertAffineTransform(detail["putar"]))[0]
    out = frame.copy()
    cv2.polylines(out, [np.round(asli).astype(np.int32)], True, (0, 255, 0), 3)
    cv2.line(out, tuple(int(v) for v in detail["p5"]), tuple(int(v) for v in detail["p17"]), _WARNA_GARIS, 3)
    return out


def susun_langkah(a, jarak=None, batas=None):
    """Return the step cards for an analysis. Each card is ready for JSON.

    A step appears only when its data exists, so a rejected photo shows the
    steps it got through. Step 8 appears only when `jarak` is given.
    """
    langkah = [{
        "no": 1, "judul": "Foto dari kamera",
        "penjelasan": "Satu frame diambil dari webcam ketika tangan sudah valid.",
        "gambar": [{"label": "Frame asli", "src": ke_data_url(a.frame, ".jpg")}],
        "angka": {"resolusi": f"{a.frame.shape[1]}×{a.frame.shape[0]}"},
    }]
    if a.tahap >= 2:
        langkah.append({
            "no": 2, "judul": "Deteksi tangan",
            "penjelasan": "MediaPipe menemukan 21 titik pada tangan. Garis hijau adalah kerangka tangan.",
            "gambar": [{"label": "21 titik", "src": ke_data_url(_gambar_landmark(a.frame, a.lm), ".jpg")}],
            "angka": {"jumlah titik": 21, "sisi": a.sisi},
        })
    if a.roi_detail is not None:
        x1, y1, x2, y2 = a.roi_detail["kotak"]
        diluruskan = a.roi_detail["rot"].copy()
        cv2.rectangle(diluruskan, (x1, y1), (x2, y2), (0, 255, 0), 3)
        sebelum = _gambar_sebelum_diluruskan(a.frame, a.roi_detail)
        # the same ruler line (index base to little-finger base) on the rotated photo: now horizontal
        M = a.roi_detail["putar"]
        p5r = tuple(int(v) for v in M @ np.array([*a.roi_detail["p5"], 1.0]))
        p17r = tuple(int(v) for v in M @ np.array([*a.roi_detail["p17"], 1.0]))
        cv2.line(diluruskan, p5r, p17r, _WARNA_GARIS, 3)
        langkah.append({
            "no": 3, "judul": "Telapak diluruskan",
            "penjelasan": ("Foto diputar sampai garis pangkal jari (garis kuning, dari pangkal telunjuk ke pangkal "
                           "kelingking) mendatar, lalu kotak telapak (hijau) dipotong dari foto yang sudah lurus. "
                           "Gambar kiri menunjukkan di mana kotak itu berada pada foto asli."),
            "gambar": [
                {"label": "Sebelum diluruskan", "src": ke_data_url(sebelum, ".jpg")},
                {"label": "Sesudah diluruskan", "src": ke_data_url(diluruskan, ".jpg")},
            ],
            "angka": {"sudut putar": f"{a.roi_detail['sudut']:.1f}°", "ukuran kotak": "1 × lebar telapak"},
        })
        potongan = a.roi_detail["potongan"]
        langkah.append({
            "no": 4, "judul": "ROI abu-abu",
            "penjelasan": ("Area tengah telapak dipotong dari foto yang sudah lurus (kiri, masih berwarna), "
                           "diubah menjadi abu-abu, lalu diperkecil ke 128×128 (kanan)."),
            "gambar": [
                {"label": "Hasil potongan (berwarna)", "src": ke_data_url(potongan), "halus": True},
                {"label": "Abu-abu 128×128", "src": ke_data_url(a.roi_detail["roi"])},
            ],
            "angka": {"ukuran potongan": f"{potongan.shape[1]}×{potongan.shape[0]}", "ukuran ROI": "128×128"},
        })
    if a.kode_detail is not None:
        kd = a.kode_detail
        langkah.append({
            "no": 5, "judul": "Masukan model",
            "penjelasan": ("Kontras garis telapak dipertegas dengan CLAHE, persis seperti saat model dilatih. Lalu gambar "
                           "disalin ke 3 saluran warna dan dinormalisasi dengan rata-rata dan simpangan ImageNet, karena "
                           "ResNet18 awalnya dilatih dengan foto ImageNet."),
            "gambar": [{"label": "Setelah CLAHE", "src": ke_data_url(kd["masukan"])}],
            "angka": {"ukuran": "128×128", "rata-rata": round(float(kd["masukan"].mean()), 1),
                      "simpangan": round(float(kd["masukan"].std()), 1)},
        })
        langkah.append({
            "no": 6, "judul": "Model CNN: peta fitur",
            "penjelasan": ("Gambar masuk ke ResNet18. Lapisan-lapisan awalnya mencari pola kecil seperti garis, lengkung, "
                           "dan tekstur kulit. Ini 8 dari 64 peta fitur di lapisan pertama: terang berarti pola yang dicari "
                           "filter itu ada di sana. Filter-filter ini dipelajari sendiri oleh model saat training, bukan "
                           "ditulis manusia."),
            "gambar": [{"label": f"Fitur {i + 1}", "src": ke_data_url(f)} for i, f in enumerate(kd["fitur"])],
            "angka": {"model": "ResNet18", "lapisan": "layer1 (64 peta, 32×32)"},
        })
        langkah.append({
            "no": 7, "judul": "Embedding 128 angka",
            "penjelasan": ("Lapisan terakhir merangkum seluruh telapak menjadi 128 angka (embedding) dengan panjang 1. "
                           "Model dilatih supaya telapak yang sama menghasilkan angka yang mirip. Angka-angka inilah yang "
                           "disimpan dan dicocokkan. Biru = negatif, merah = positif."),
            "gambar": [{"label": "128 angka (8×16)", "src": ke_data_url(gambar_embedding(kd["embedding"]))}],
            "angka": {"dimensi": int(np.asarray(kd["embedding"]).size), "versi model": a.model_versi},
            "legenda": [{"arah": "negatif", "warna": WARNA_NEGATIF}, {"arah": "nol", "warna": WARNA_NOL},
                        {"arah": "positif", "warna": WARNA_POSITIF}],
        })
    if jarak is not None:
        terkecil = min((d["jarak"] for d in jarak), default=None)
        langkah.append({
            "no": 8, "judul": "Pencocokan",
            "penjelasan": ("Embedding scan dibandingkan dengan 8 embedding setiap tangan terdaftar memakai jarak kosinus "
                           "(0 = arahnya sama persis). Jarak sebuah tangan adalah jarak ke sampelnya yang paling mirip. "
                           "Jarak terkecil di bawah batas berarti dikenali."),
            "gambar": [],
            "angka": {
                "jarak terkecil": "-" if terkecil is None else round(terkecil, 3),
                "batas": "-" if batas is None else round(batas, 2),
            },
            "jarak": jarak,
        })
    return langkah
