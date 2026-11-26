"""model.py - tempat model pengenal telapak buatanmu.

Tugas model: mengubah satu ROI telapak (gambar abu-abu 128x128) menjadi daftar angka
yang disebut embedding. Telapak yang sama harus menghasilkan angka yang mirip.

Sekarang model BELUM ADA. Selama NAMA_MODEL masih None:
- Daftar dan Scan tetap mendeteksi tangan dan memotong ROI,
- lalu berhenti dengan pesan "Model belum dibuat".

Cara memasang modelmu nanti:
1. Latih modelmu.
2. Isi NAMA_MODEL, misalnya "resnet18-v1".
3. Isi fungsi buat_embedding supaya mengembalikan list angka.
"""

NAMA_MODEL = None


def model_siap():
    return NAMA_MODEL is not None


def buat_embedding(roi):
    """roi: numpy array abu-abu 128x128.

    Kembalikan list angka, misalnya [0.12, -0.05, 0.31, ...].
    """
    raise NotImplementedError("Model belum dibuat. Isi fungsi buat_embedding di backend/model.py")
