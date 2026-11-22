# Telapak AI: Pengenalan Identitas dari Telapak Tangan dengan Model CNN

Aplikasi web: seseorang **mendaftarkan telapak tangannya**, lalu sistem **mengenalinya kembali** lewat webcam. Pengenalnya adalah **model CNN (ResNet18 + ArcFace) yang dilatih sendiri** pada dataset Tongji Contactless Palmprint.

Penjelasan cara kerja untuk presentasi ada di [docs/penjelasan.md](docs/penjelasan.md). Proyek ini lanjutan dari Telapak ID (versi Canny/Gabor). Untuk membandingkan dengan metode klasik, jalankan proyek lama itu.

---

## 1. Isi folder

```
backend/                  Flask API + computer vision (Python)
  app.py                  endpoint API, pencocokan jarak kosinus
  model.py                model CNN: ResNet18 → embedding 128 angka, kepala ArcFace, Pengenal
  palm.py                 deteksi tangan (MediaPipe) dan pemotongan ROI telapak 128×128
  validation.py           aturan tangan: jari terbuka, telapak menghadap kamera, kualitas ROI
  pipeline.py             menjalankan langkah 1–7 untuk satu foto
  steps.py                kartu langkah untuk halaman hasil
  db.py                   model MySQL (orang, telapak) dan fungsi data
  tools/ambil_foto.py     mengambil foto telapak dari webcam (untuk dataset sendiri)
training/                 menyiapkan data, melatih, dan mengevaluasi model
  data.py                 dataset Tongji → ROI 128×128 + pembagian latih/validasi/uji per telapak
  latih.py                training ResNet18 + ArcFace
  evaluasi.py, metrik.py  EER, rank-1, ROC, histogram jarak
frontend/                 React + Tailwind + Vite
docs/                     rancangan, rencana, dan penjelasan cara kerja
hasil/                    grafik dan angka hasil training/evaluasi (untuk laporan)
data/                     dataset dan ROI (tidak ikut git)
model/                    model hasil training, telapak.pt (tidak ikut git)
```

## 2. Kebutuhan

- Windows 10/11, Python 3.10, Node.js 20+, MySQL 8 (Laragon).
- GPU NVIDIA disarankan untuk training (diuji di RTX 3050 6 GB, sekitar 10 menit). Aplikasi bisa berjalan di CPU.
- `mediapipe` dikunci ke **0.10.14** (butuh antarmuka `mp.solutions`), `numpy<2`.

```bash
C:\laragon\bin\python\python-3.10\python.exe -m venv .venv
.venv\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -r backend\requirements.txt
```

## 3. Melatih model

Unduh *Original Images* dari [Tongji Contactless Palmprint Dataset](https://cslinzhang.github.io/ContactlessPalm) dan ekstrak sampai ada folder `session1/` dan `session2/`. Lalu dari folder proyek:

```bash
.venv\Scripts\python -m training.data --asli "D:/Original Images" --keluar data/roi     # sekitar 20 menit
.venv\Scripts\python -m training.latih --roi data/roi --epoch 20                         # sekitar 10 menit di GPU
.venv\Scripts\python -m training.evaluasi --sendiri data/sendiri                          # angka akhir + grafik
```

- `training.data` memotong ROI dengan **fungsi yang sama dengan aplikasi** (`palm.ambil_roi`), lalu menulis `data/pembagian.json` dan `hasil/siapkan_data.txt`.
- `training.latih` menyimpan epoch dengan EER validasi terbaik ke `model/telapak.pt`, dan grafik ke `hasil/grafik_training.png`.
- `training.evaluasi` memilih batas dari data validasi (disimpan ke file model), lalu mengukur EER dan rank-1 pada data uji. Opsi `--sendiri` juga menguji foto webcam sendiri (satu folder per tangan).
- Setiap training menghasilkan **versi model** baru. Tangan yang terdaftar dengan versi lama harus didaftar ulang.

## 4. Menjalankan aplikasi

Nyalakan MySQL di Laragon, pastikan `model/telapak.pt` sudah ada, lalu:

```bash
# Backend
cd backend
copy .env.example .env               # DATABASE_URL ke database telapak_ai
..\.venv\Scripts\python app.py       # http://127.0.0.1:5000

# Frontend (terminal lain)
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

Database `telapak_ai` dibuat otomatis. Kalau MySQL mati atau model belum ada, backend menampilkan pesan yang jelas lalu berhenti.

## 5. Alur pemakaian

| Halaman | Isi |
|---|---|
| **Scan** (`/`) | Kamera dengan kerangka tangan dan kotak telapak. Foto diambil otomatis saat semua syarat terpenuhi. Slider batas mengatur ambang kecocokan. Bawaannya 0,50 dari `BATAS` di `backend/.env`; kosongkan untuk memakai batas model (0,619). |
| **Hasil** (`/hasil`) | 8 langkah pemrosesan dengan gambar, lalu hasil: dikenali, tidak dikenal, atau gagal. |
| **Daftar** (`/daftar`) | Data diri, sisi tangan, persetujuan, lalu 8 sampel dari kamera. |
| **Data terdaftar** (`/data`) | Daftar orang dan versi model setiap tangan; bisa dihapus. |

## 6. Proses per foto

| # | Tahap | Keterangan |
|---|---|---|
| 1 | Foto dari kamera | Satu frame saat tangan valid |
| 2 | Deteksi tangan | MediaPipe Hands, 21 titik |
| 3 | Telapak diluruskan | Diputar sampai garis pangkal telunjuk–kelingking mendatar |
| 4 | ROI abu-abu | Kotak selebar telapak di bawah pangkal jari, diperkecil ke 128×128 |
| 5 | Masukan model | CLAHE, 3 saluran, normalisasi ImageNet |
| 6 | Model CNN | ResNet18 hasil fine-tuning; halaman hasil menampilkan 8 peta fitur lapisan pertama |
| 7 | Embedding | 128 angka dengan panjang 1 |
| 8 | Pencocokan | Jarak kosinus ke 8 sampel tiap tangan terdaftar, ambil yang terdekat, bandingkan dengan batas |

## 7. Hasil

Pada 85 telapak uji Tongji yang tidak pernah dilihat model (daftar dengan sesi 1, scan dengan sesi 2): **EER 3,98%, rank-1 86,6%**. Pada foto webcam sendiri: rank-1 97,9%, EER 10,4%. Rincian dan grafik ada di `hasil/` dan [docs/penjelasan.md](docs/penjelasan.md).

## 8. Validasi, privasi, dan log

- Validasi tangan dicek di browser dan di server: tangan terdeteksi, kelima jari terbuka, telapak (bukan punggung) menghadap kamera, jarak pas, tangan diam, dan ROI tidak terlalu gelap, terang, atau buram.
- Yang disimpan hanya 8 embedding × 128 angka per tangan beserta versi model. **Foto tidak disimpan.** Persetujuan wajib dicentang, dan data bisa dihapus dari halaman *Data terdaftar*.
- Terminal backend mencatat setiap analisis (`analisis ... model=<versi> hasil=lolos`), isi data yang disimpan (`tersimpan: ... 4096 byte = 8 sampel x 128 angka`), dan jarak ke setiap tangan saat scan (`dibandingkan: ... per sampel: ...`).

## 9. Menjalankan tes

```bash
cd backend && ..\.venv\Scripts\python -m pytest -q     # API, database, pipeline, model, validasi
.venv\Scripts\python -m pytest training/tests -q       # persiapan data, metrik, training singkat
cd frontend && npm test && npm run build
```

Tes backend memakai SQLite di memori dan pengenal tiruan, jadi tidak membutuhkan MySQL maupun file model.

## 10. Keterbatasan

- Model hanya dilatih dengan foto Tongji (kamera industri, lampu tetap). Di webcam akurasinya lebih rendah (*domain shift*). Pengumpulan foto sendiri dengan `backend/tools/ambil_foto.py` untuk fine-tuning tambahan adalah langkah berikutnya.
- Sekitar 20% foto Tongji tidak bisa dipotong (tangan tidak terdeteksi di foto gelap, atau kotak keluar dari foto).
- Belum ada anti-pemalsuan. Ini prototipe pembelajaran, belum layak sebagai pengaman utama.

## 11. Rujukan

- Zhang, L., Li, L., Yang, A., Shen, Y., & Yang, M. (2017). Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach. *Pattern Recognition*, 69, 199–212. (dataset)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*. (ResNet)
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. *CVPR*. (ArcFace)

Periksa kembali detail rujukan di Google Scholar sebelum dikutip di laporan.
