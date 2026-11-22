# Rancangan: Telapak AI (pengenalan telapak dengan model yang dilatih)

Tanggal: 22 November 2026
Asal: proyek baru, dibangun dari bagian yang berguna di `Desktop/telapak-id` (proyek lama tidak diubah).

> **Tenggat tugas: 23 November 2026 malam.** Bagian 9 di bawah memangkas rancangan ini. Kalau ada yang bertentangan, ikuti bagian 9.

## 1. Tujuan dan alasannya

Tugas akhir mata kuliah **Pengolahan Citra** mewajibkan computer vision **dengan model yang dilatih sendiri**.
Telapak ID yang lama mengenali orang dengan aturan buatan manusia (Canny/Gabor ditambah rumus jarak), tanpa
training. Proyek baru ini mengganti tahap pengenal dengan **model CNN yang di-fine-tune** pada dataset telapak,
dan alur aplikasinya tetap sama: daftarkan telapak seseorang, lalu kenali dia lagi lewat webcam dari garis tangannya.

Pemilik proyek masih pemula di AI. Karena itu, setiap tahap harus bisa ia jelaskan sendiri kepada dosen.
Pengerjaannya dipandu langkah demi langkah, dan setiap langkah memberi hasil yang bisa dilihat (angka, grafik, atau halaman).

Dikatakan berhasil kalau:
- Ada model yang dilatih dari dataset, dengan grafik training dan angka evaluasi untuk laporan.
- Aplikasi web bisa mendaftarkan dan mengenali orang memakai model itu.
- Ada perbandingan yang jujur antara model dan metode klasik (Gabor/Canny) pada data yang sama.

## 2. Sasaran, bukan sasaran, dan yang harus dihindari

**Sasaran**
- Pipeline training yang bisa dijalankan ulang di laptop pemilik (RTX 3050 6 GB).
- Model yang mengubah potongan telapak 128×128 menjadi **128 angka (embedding)**. Telapak yang sama menghasilkan embedding yang berdekatan.
- Aplikasi web (daftar, scan, hasil, data terdaftar) memakai embedding tersebut.
- Evaluasi pada telapak yang **tidak pernah dilihat saat training**: EER, kurva ROC, dan akurasi identifikasi (rank-1).
- Pembanding: Gabor dan Canny pada data uji yang sama.

**Bukan sasaran (untuk sekarang)**
- YOLO atau melatih detektor tangan. MediaPipe tetap dipakai untuk mencari tangan.
- Deteksi kesehatan (hemoglobin dan sejenisnya).
- Anti-pemalsuan (foto telapak di layar), keamanan produksi, atau deploy ke internet.
- Mengumpulkan dataset sendiri. Ini menyusul; alat pengambil foto ikut dibawa supaya langkah ini nanti mudah.

**Yang harus dihindari (dianggap gagal walaupun "jalan")**
- Model yang harus **dilatih ulang setiap ada orang baru** mendaftar.
- **Kebocoran data:** telapak yang sama muncul di data latih dan data uji, sehingga akurasinya terlihat tinggi palsu.
- Kode atau konsep yang terlalu rumit untuk dijelaskan pemilik di presentasi.
- Mencampur embedding dari dua versi model yang berbeda di database.

## 3. Batasan

- Windows 11, Python 3.10, GPU NVIDIA RTX 3050 Laptop 6 GB, MySQL lewat Laragon.
- `mediapipe==0.10.14` dikunci (antarmuka `mp.solutions`), `numpy<2`.
- Dataset publik hanya untuk penelitian: tidak boleh di-commit ke git dan makalahnya wajib dikutip di laporan.
- Foto webcam berbeda dari foto dataset (kamera, lampu, resolusi). Karena itu dipakai augmentasi, dan pengujian pada foto `data_uji` milik pemilik wajib dilakukan.

## 4. Cara yang dipertimbangkan

| Cara | Isi | Untung / rugi |
|---|---|---|
| **A. Fine-tuning model pretrained** (dipilih) | ResNet18 yang sudah dilatih dengan ImageNet dilatih lagi dengan telapak | Paling akurat untuk data terbatas, training sekitar 30–60 menit di laptop, dan referensinya banyak. Pemilik perlu memahami konsep pretrained |
| B. CNN kecil dari nol | Rancangan lapisan sendiri, bobot awal acak | Mudah dijelaskan per lapisan, tapi akurasinya biasanya lebih rendah dan butuh banyak coba-coba |
| C. SVM dengan fitur Gabor (dicoret) | Classifier klasik | Menghafal daftar orang, sehingga harus dilatih ulang setiap ada pendaftar baru; bertentangan dengan alur aplikasi |

Pemilik memilih **A** pada 22 November 2026, setelah membandingkannya dengan B dan menanyakan hubungannya dengan
YOLO. YOLO juga di-fine-tune, tetapi tugasnya mencari letak benda, bukan menentukan pemiliknya, jadi tidak dipakai di sini.

## 5. Keputusan pemilik

- Proyek dibuat baru di folder sendiri. Hanya bagian yang berguna yang diambil dari repo lama, dan repo lama tidak diubah.
- Cara A: fine-tuning.
- Alur aplikasi sama dengan yang lama: daftar dengan 8 sampel, lalu scan untuk mengenali.
- Dataset publik dulu; dataset sendiri menyusul.
- Pemilik dipandu satu langkah demi satu langkah.

## 6. Rancangan

### 6.1 Alur

```
TRAINING (sekali, di laptop):
dataset publik → MediaPipe → potong ROI 128×128 (kode yang sama dengan aplikasi) → simpan ROI
→ latih ResNet18 + kepala ArcFace → simpan model → evaluasi pada telapak yang belum pernah dilihat → grafik

APLIKASI (setiap daftar/scan):
webcam → MediaPipe → validasi → ROI 128×128 → model → embedding 128 angka
→ daftar: simpan 8 embedding | scan: jarak kosinus ke semua tangan, ambil sampel terdekat, bandingkan dengan batas
```

ROI dataset dan ROI webcam **dipotong dengan fungsi yang sama** (`ambil_roi` dari `palm.py`), jadi model melihat potongan yang seragam.

### 6.2 Data

- Dataset utama: **Tongji Contactless Palmprint** (600 telapak × 20 foto, 2 sesi). Dataset kedua, kalau pemilik sudah mengunduhnya: **Tongji MPD** (foto ponsel, lebih mirip webcam). IITD menyusul kalau formulirnya disetujui.
- Setiap telapak (tangan kiri dan kanan dihitung terpisah) adalah satu "identitas".
- **Pembagian berdasarkan identitas:** 80% telapak untuk latih, 20% untuk uji. Telapak di data uji tidak pernah dilihat model, sama seperti orang baru yang mendaftar di aplikasi. Pembagian dibuat sekali dengan seed tetap lalu disimpan ke file, supaya hasilnya bisa diulang.
- Langkah pertama yang wajib: jalankan MediaPipe pada sebagian foto dataset dan laporkan berapa persen yang berhasil dipotong. Kalau persentasenya rendah (misalnya foto Tongji abu-abu atau jarinya terpotong), diputuskan bersama pemilik apakah tetap memakai ROI dari MediaPipe atau memakai ROI bawaan dataset.
- Augmentasi saat training: kecerahan dan kontras, putaran ±10°, geseran kecil, blur ringan, noise. Tujuannya meniru kondisi webcam.
- Foto dataset dan ROI hasil potongan disimpan di `data/`, yang masuk `.gitignore`.

### 6.3 Model dan training

- PyTorch + torchvision, `resnet18` dengan bobot ImageNet. Masukan ROI abu-abu disalin ke 3 kanal, lalu dinormalisasi seperti ImageNet.
- Lapisan terakhir diganti dengan lapisan yang menghasilkan **embedding 128 angka**, yang kemudian dinormalisasi (panjangnya 1).
- Saat training, di atas embedding dipasang kepala **ArcFace**: model belajar menebak "telapak nomor berapa" di antara telapak data latih, dengan margin supaya kelompok tiap telapak terpisah rapi. **Setelah training, kepala ini dibuang**. Yang dipakai aplikasi hanya embedding, sehingga orang baru tidak memerlukan training ulang. Penjelasan ini juga yang dibawakan pemilik di presentasi.
- Pengaturan awal: optimizer AdamW, sekitar 30 epoch, batch 64, learning rate dengan cosine schedule. Nilai pastinya diserahkan ke pembuat dan dicatat di file konfigurasi.
- Setiap epoch dicatat: loss latih, akurasi latih, dan EER pada data uji. Hasilnya disimpan sebagai grafik PNG untuk laporan.
- Keluaran: `model/telapak.pt` berisi bobot, ukuran embedding, **versi model**, dan batas yang disarankan. File ini tidak di-commit, tapi cara membuatnya terdokumentasi.

### 6.4 Evaluasi (bahan laporan)

Pada telapak uji: sesi 1 dipakai sebagai "pendaftaran" (8 foto), sesi 2 sebagai "scan".
- **EER** dan **kurva ROC**.
- **Rank-1:** berapa persen scan yang telapak terdekatnya memang telapak yang benar.
- Histogram jarak untuk pasangan "sama" dan "beda", lalu pemilihan batas.
- Tabel pembanding pada data yang sama: **model vs Gabor vs Canny+arah**. Kode Gabor/Canny dibawa dari repo lama khusus untuk evaluasi ini.
- Uji tambahan pada foto `data_uji` milik pemilik: apakah tangan kanan dan kiri pemilik terpisah dengan baik.

### 6.5 Aplikasi

Diambil dari repo lama lalu disederhanakan:
- **Dibawa:** frontend React (kamera berpanduan, halaman Scan, Hasil, Daftar, Data terdaftar, Detail), deteksi tangan MediaPipe di browser, `validation.py`, potongan ROI dari `palm.py`, `pipeline.py`, `db.py`, `app.py`, `tools/ambil_foto.py`, beserta tes yang masih relevan.
- **Berubah:** tahap 6–7 di halaman hasil menjadi "Model CNN → embedding" (dengan visualisasi embedding). Pilihan metode dan ukuran kotak ROI dihapus dari aplikasi (kotak tetap 1.0).
- **Database:** tabel `telapak` menyimpan 8 embedding × 128 float32 (4 KB) dan `model_versi`. Saat scan, hanya tangan dengan versi model yang sama yang dibandingkan. Kalau model dilatih ulang, aplikasi meminta pendaftaran ulang.
- **Pencocokan:** jarak kosinus, sampel terdekat dari 8 sampel, lalu dibandingkan dengan batas dari hasil evaluasi.

### 6.6 Struktur folder (usulan)

```
telapak-ai/
  docs/            rancangan ini, rencana, catatan langkah
  training/        siapkan_data.py, dataset.py, model.py, latih.py, evaluasi.py, baseline/ (Gabor, Canny)
  backend/         Flask API, memuat model/telapak.pt
  frontend/        React
  data/            dataset dan ROI (tidak di-commit)
  model/           hasil training (tidak di-commit)
  hasil/           grafik dan tabel evaluasi untuk laporan
```

### 6.7 Cara memandu pemilik

- Setiap langkah di rencana punya: tujuan dengan bahasa sederhana, apa yang dikerjakan pemilik (unduh, jalankan perintah, lihat hasil), apa yang dikerjakan Claude (kode), dan **hasil yang harus terlihat** sebelum lanjut ke langkah berikutnya.
- Lingkungan Python terpisah (`venv`) di folder proyek, supaya PyTorch tidak mengganggu proyek lama.
- Repo git baru dengan riwayat sendiri.

## 7. Diserahkan ke pembuat

- Nilai hyperparameter yang tepat, isi augmentasi, dan bentuk visualisasi embedding.
- Detail kode backend dan frontend, selama alur dan tampilannya sama dengan aplikasi lama.
- Format tabel dan grafik evaluasi.

Keputusan kecil yang diambil Claude (pemilik boleh mengubahnya):
- Backend menjalankan model di CPU kalau GPU tidak tersedia; satu foto cukup cepat untuk diproses di CPU.
- Foto `data_uji` milik pemilik disalin ke `data/sendiri/` di proyek baru (tidak di-commit, karena termasuk data biometrik).
- Kalau Tongji dan MPD sama-sama tersedia, keduanya digabung untuk training. Hasil evaluasi dilaporkan per dataset.

## 8. Risiko yang sudah diketahui

- MediaPipe mungkin gagal pada sebagian foto Tongji. Ini dicek di langkah pertama (lihat 6.2).
- Akurasi pada webcam bisa lebih rendah daripada pada data uji Tongji karena kondisi fotonya berbeda. Ini diukur dan ditulis apa adanya di laporan; tambahan dataset sendiri nanti membantu.
- Instalasi PyTorch dengan CUDA di Windows kadang bermasalah. Cadangannya Google Colab untuk training saja.

## 9. Revisi karena tenggat (berlaku di atas bagian lain)

Pemilik menjawab pada 22 November 2026: tenggatnya **besok malam**, dan dosen hanya meminta **hasil jadi yang bisa didemokan
serta penjelasan cara kerjanya dari pemilik** (tidak wajib notebook). Karena itu:

1. **Clone, bukan tulis ulang.** `telapak-ai` dibuat dengan `git clone` dari `telapak-id`, lalu diubah di sana. Proyek
   lama tetap utuh (alasan awal pemilik terpenuhi), dan aplikasi yang sudah jalan tidak perlu dibangun ulang. Bagian yang
   tidak dipakai lagi baru dibersihkan kalau masih ada waktu.
2. **Model menjadi satu-satunya metode** (keputusan pemilik, 22 November 2026). Canny, Gabor, dan pilihan `METODE` dihapus
   dari proyek baru: `canny.py`, `metode.py`, bagian Gabor/Competitive Code di `palm.py` (potongan ROI tetap dipakai),
   `tools/bandingkan_metode.py`, `tools/bandingkan_skala.py`, `tools/lihat_kode.py`, beserta tesnya. Ukuran kotak ROI tetap
   1.0 (`ROI_SKALA` dihapus). Kode yang disimpan berupa 8 embedding float32 × 128 per tangan, dengan `model_versi`.
   Penyesuaian yang perlu: validasi sampel di `app.py`, kolom di `db.py`, tahap 6–7 di `steps.py` (CNN → embedding), dan
   skala grafik jarak di frontend. Untuk membandingkan dengan Canny/Gabor, pemilik menjalankan proyek lama `telapak-id`.
3. **Dataset: hanya Tongji Contactless.** Dataset ini bisa diunduh langsung dari Google Drive, sebagai *Original Images*
   dan *ROI Images*. MPD ada di Baidu Pan (butuh akun dan kode ekstraksi), dan IITD butuh persetujuan email; keduanya terlalu
   lambat untuk tenggat ini. Struktur Tongji: `session1/` dan `session2/`, dengan nama file sama untuk telapak yang sama,
   `00001–00010` = telapak 1, dan seterusnya (600 telapak × 10 foto per sesi).
4. **ROI:** coba MediaPipe dan `ambil_roi` pada foto asli. Kalau hasilnya kurang dari sekitar 80% berhasil, pakai ROI bawaan
   Tongji untuk training, dan tutup perbedaannya dengan augmentasi yang lebih kuat. Keputusan ini dicatat di laporan.
5. **Training dipersingkat:** sekitar 15–20 epoch, tanpa pencarian hyperparameter.
6. **Evaluasi minimum:** EER, rank-1, grafik loss/akurasi per epoch, kurva ROC, dan histogram jarak pada telapak uji.
   Tidak ada pembanding Gabor/Canny di proyek ini (bagian 6.4 yang menyebutnya tidak berlaku).
7. **Ditambahkan: `docs/penjelasan.md`.** Penjelasan cara kerja dengan bahasa sederhana (apa itu model, pretrained,
   fine-tuning, embedding, ArcFace, EER, alasan pembagian data per telapak) beserta angka hasil training. Dokumen ini
   bahan pemilik untuk menjelaskan ke dosen.
8. **Ditunda:** dataset sendiri, notebook, struktur folder di 6.6 (cukup tambah `training/`, `data/`,
   `model/`, dan `hasil/`).
9. **Cadangan:** kalau PyTorch dengan CUDA gagal dipasang dalam sekitar 30 menit, training dipindah ke Google Colab
   (GPU gratis), lalu file model diunduh ke laptop.
