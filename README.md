# Telapak AI

Aplikasi web untuk **mendaftarkan telapak tangan** lewat webcam, lalu **mengenali pemiliknya** saat
telapak di-scan lagi.

> **Status: model belum ada.** Model pengenal (CNN) sedang dibuat ulang dari nol. Aplikasi sudah bisa
> menyalakan kamera, mendeteksi tangan, mengecek posisinya, dan memotong bagian telapak (ROI). Setelah
> itu Daftar dan Scan berhenti dengan pesan **"Model belum dibuat"** sampai `backend/model.py` diisi.
> Versi lama yang memakai ResNet18 + ArcFace masih ada di branch `main`.

## 1. Cara kerja singkat

```
Webcam -> MediaPipe (21 titik tangan) -> cek tangan -> potong ROI 128x128 -> MODEL -> embedding (daftar angka)
                                                                                         |
                                     Daftar: simpan 8 embedding per tangan ke MySQL  <---+
                                     Scan  : bandingkan dengan embedding tersimpan   <---+
```

- **Daftar**: kamera mengambil 8 foto. Setiap foto diubah model menjadi embedding. Kedelapan embedding
  disimpan bersama nama orangnya. Foto tidak disimpan.
- **Scan**: satu foto diubah menjadi embedding, lalu dibandingkan dengan semua tangan terdaftar memakai
  jarak kosinus (0 = sama persis). Jarak terkecil di bawah **batas** berarti dikenali.

## 2. Isi folder

```
backend/                Flask API (Python)
  app.py                semua endpoint API
  model.py              TEMPAT MODEL: isi NAMA_MODEL dan fungsi buat_embedding di sini
  palm.py               deteksi tangan (MediaPipe) dan potong ROI telapak 128x128
  validation.py         aturan tangan: jari terbuka, telapak menghadap kamera, kualitas ROI
  pipeline.py           memproses satu foto langkah demi langkah
  pencocokan.py         jarak kosinus antara embedding
  steps.py              kartu langkah untuk halaman Hasil
  db.py                 tabel MySQL (orang, tangan) dan fungsi datanya
  openapi.yaml          dokumentasi API (OpenAPI 3)
  tests/                tes otomatis (pytest)
frontend/               React + Tailwind + Vite
  src/pages/            halaman Scan, Hasil, Daftar, Data terdaftar, Detail orang
  src/components/       kamera (PalmCamera), grafik jarak, komponen tampilan
  src/lib/              panggilan API dan aturan tangan di browser
data/                   dataset (tidak ikut git)
```

## 3. Kebutuhan

- Windows 10/11, Python 3.10, Node.js 20+, MySQL 8 (Laragon).
- `mediapipe` dikunci ke **0.10.14** dan `numpy<2`.

```bash
C:\laragon\bin\python\python-3.10\python.exe -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
```

## 4. Menjalankan aplikasi

Nyalakan MySQL di Laragon, lalu:

```bash
# Backend (terminal 1)
cd backend
copy .env.example .env
..\.venv\Scripts\python app.py        # http://127.0.0.1:5000

# Frontend (terminal 2)
cd frontend
npm install
npm run dev                           # http://localhost:5173
```

Database `telapak_ai` dan tabelnya dibuat otomatis. Kalau MySQL mati, backend menampilkan pesan lalu
berhenti.

> Database dari versi lama punya tabel `telapak` yang tidak dipakai lagi (datanya dibuat model lama).
> Versi ini memakai tabel baru `tangan`. Tabel lama boleh dihapus: `DROP TABLE telapak;`

## 5. Dokumentasi API

Setelah backend jalan, buka **http://127.0.0.1:5000/api/docs** (Swagger UI). Semua endpoint bisa dicoba
langsung dari halaman itu. Spesifikasinya ada di [backend/openapi.yaml](backend/openapi.yaml).

| Endpoint                     | Kegunaan                               |
| ---------------------------- | -------------------------------------- |
| `GET /api/pengaturan`        | status model dan batas bawaan slider   |
| `POST /api/sampel`           | proses satu foto sampel saat mendaftar |
| `POST /api/scan`             | kenali pemilik telapak dari satu foto  |
| `GET/POST/DELETE /api/orang` | daftar, simpan, atau hapus semua orang |
| `GET/DELETE /api/orang/{id}` | detail atau hapus satu orang           |

## 6. Memasang model nanti

Buka [backend/model.py](backend/model.py). Aplikasi hanya butuh dua hal:

```python
NAMA_MODEL = "resnet18-v1"   # nama apa saja; disimpan bersama setiap tangan yang didaftarkan

def buat_embedding(roi):     # roi = gambar abu-abu 128x128 (numpy array)
    ...
    return [0.12, -0.05, ...]  # list angka (embedding)
```

- Telapak yang sama harus menghasilkan embedding yang mirip (jarak kosinus kecil).
- Panjang embedding bebas, asal selalu sama untuk model yang sama.
- Tangan yang didaftarkan dengan model lain tidak ikut dibandingkan. Setelah ganti model, daftar ulang.
- Untuk membandingkan 2 model, ganti `NAMA_MODEL` dan `buat_embedding`, lalu daftarkan ulang.

## 7. Tes dan format kode

```bash
cd backend && ..\.venv\Scripts\python -m pytest -q     # tes backend (pakai SQLite, tidak butuh MySQL)
cd frontend && npm test                                # tes aturan tangan di browser
```

Format kode dirapikan dengan **Prettier** (JavaScript, CSS, OpenAPI, README) dan **Black** (Python):

```bash
cd frontend && npm run format                # Prettier
cd backend && ..\.venv\Scripts\python -m black .   # Black
```

Kalau Windows memblokir Black ("Application Control policy"), pasang versi tanpa DLL:
`..\.venv\Scripts\python -m pip install --no-binary black black`.

## 8. Privasi

Yang disimpan hanya embedding (daftar angka) per tangan, ditambah nama, umur, dan status. Foto tidak
disimpan. Pendaftaran wajib mencentang persetujuan, dan data bisa dihapus dari halaman _Data terdaftar_.
