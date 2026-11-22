# Telapak AI: Cara Kerja

Bahan untuk menjelaskan proyek ini kepada dosen. Setiap istilah dijelaskan dengan bahasa sederhana, lalu ditunjukkan di bagian mana kodenya berada.

---

## 1. Ringkasan dalam satu paragraf

Telapak AI mengenali seseorang dari **garis telapak tangannya** lewat webcam. Prosesnya: MediaPipe menemukan tangan, lalu area tengah telapak dipotong menjadi gambar 128×128. Gambar itu dimasukkan ke **model CNN (ResNet18)** yang **saya latih sendiri** dengan dataset Tongji (12.000 foto dari 600 telapak). Model mengubah setiap telapak menjadi **128 angka (embedding)**. Saat mendaftar, 8 embedding disimpan ke MySQL. Saat scan, embedding baru dibandingkan dengan yang tersimpan: kalau jaraknya di bawah batas, orang itu dikenali.

```
                 ┌──────────── sekali, di laptop (training) ─────────────┐
 Dataset Tongji → MediaPipe → potong ROI 128×128 → latih ResNet18 + ArcFace → model/telapak.pt
                 └───────────────────────────────────────────────────────┘

                 ┌──────────── setiap daftar / scan (aplikasi) ──────────┐
 Webcam → MediaPipe → validasi → ROI 128×128 → model → 128 angka → simpan (daftar) / cocokkan (scan)
                 └───────────────────────────────────────────────────────┘
```

---

## 2. Istilah dasar

**Model AI.** Sebuah fungsi besar berisi jutaan angka yang bisa disetel, disebut **bobot** (*weights*). ResNet18 punya sekitar 11 juta bobot. Gambar masuk, dikalikan dan dijumlahkan dengan bobot-bobot itu lapis demi lapis, lalu keluar hasilnya (di sini: 128 angka).

**Training (melatih).** Awalnya bobot belum cocok untuk telapak. Training artinya:
1. tunjukkan sekelompok foto (satu *batch*, misalnya 64 foto) ke model,
2. hitung seberapa salah jawabannya (**loss**),
3. setel semua bobot sedikit ke arah yang mengurangi kesalahan (**backpropagation** + **optimizer** AdamW),
4. ulangi untuk seluruh dataset. Satu putaran penuh disebut **satu epoch**. Kita melakukan sekitar 20 epoch.

**CNN (Convolutional Neural Network).** Jenis model untuk gambar. Lapisan-lapisannya menggeser filter kecil di atas gambar untuk mencari pola. Lapisan awal menemukan garis dan tepi, lapisan tengah menemukan lengkungan dan tekstur, lapisan akhir merangkum semuanya. Bedanya dengan Gabor/Canny di proyek lama: **filter-filter ini tidak ditulis manusia, tetapi dipelajari sendiri oleh model dari data.**

**Pretrained dan fine-tuning.** ResNet18 sudah pernah dilatih oleh peneliti dengan **ImageNet** (sekitar 1,2 juta foto umum: kucing, mobil, dan sebagainya), jadi sudah "bisa melihat" garis dan tekstur. Kita tidak mulai dari nol. Kita **melanjutkan training** dengan foto telapak supaya model belajar membedakan telapak. Ini disebut **fine-tuning**. Hasilnya jauh lebih bagus daripada mulai dari bobot acak, karena data telapak kita terbatas.

---

## 3. Bagian-bagian sistem

### 3.1 Mencari tangan dan memotong telapak (bukan AI yang kita latih)

`backend/palm.py`, fungsi `ambil_roi_detail`
1. **MediaPipe Hands** (model buatan Google, sudah jadi) memberi 21 titik tangan.
2. Garis dari pangkal telunjuk (titik 5) ke pangkal kelingking (titik 17) dipakai sebagai penggaris: foto diputar sampai garis ini mendatar.
3. Kotak persegi selebar telapak dipotong di bawah garis itu, diubah ke abu-abu, lalu diperkecil menjadi **128×128**. Ini disebut **ROI** (*Region of Interest*).

**Fungsi yang sama dipakai untuk memotong foto dataset saat training** (`training/data.py`). Jadi model dilatih dengan potongan yang persis sama dengan yang nanti datang dari webcam.

### 3.2 Pengolahan sebelum masuk model

`backend/model.py`, fungsi `siapkan_masukan`
- **CLAHE**: kontras garis telapak dipertegas (teknik pengolahan citra klasik).
- Nilai piksel dibagi 255, disalin ke 3 saluran (R, G, B) karena ResNet18 menerima gambar berwarna, lalu dinormalisasi dengan rata-rata dan simpangan baku ImageNet.

### 3.3 Model: ResNet18 → embedding 128 angka

`backend/model.py`, kelas `JaringanTelapak`
- Badan model: **ResNet18** pretrained ImageNet.
- Lapisan terakhir ResNet (yang aslinya menebak 1000 kelas ImageNet) diganti dengan lapisan baru yang mengeluarkan **128 angka**.
- 128 angka itu dinormalisasi sehingga panjang vektornya 1. Inilah **embedding**: "sidik angka" dari sebuah telapak.

### 3.4 Cara model belajar: ArcFace

`backend/model.py`, kelas `ArcFace`
Selama training saja, di atas embedding dipasang lapisan **ArcFace** yang membuat model menebak "ini telapak nomor berapa" (dari 420 telapak data latih). ArcFace memberi **margin**: tebakan yang benar baru dianggap cukup kalau embedding-nya sangat dekat dengan pusat kelompok telapaknya. Akibatnya, embedding dari telapak yang sama berkumpul rapat, dan telapak yang berbeda saling menjauh.

**Setelah training, kepala ArcFace dibuang.** Aplikasi hanya memakai embedding. Karena itu **orang baru bisa mendaftar tanpa melatih ulang model**. Model tidak menghafal daftar orang, tetapi sudah belajar *cara* membedakan telapak, mirip sistem pengenalan wajah di ponsel.

### 3.5 Mendaftar dan mengenali

`backend/app.py`
- **Daftar:** 8 foto diambil, masing-masing dijadikan embedding, lalu ke-8 embedding (8 × 128 angka = 4 KB) disimpan di MySQL bersama **versi model**. Foto tidak disimpan.
- **Scan:** embedding foto baru dibandingkan dengan semua tangan terdaftar memakai **jarak kosinus** = 1 − (embedding₁ · embedding₂). Jarak 0 berarti arahnya sama persis. Jarak ke sebuah tangan = jarak ke sampelnya yang paling mirip. Kalau jarak terkecil **di bawah batas**, orang itu dikenali.
- Kalau model dilatih ulang, versi model berubah dan tangan lama tidak dibandingkan lagi (harus daftar ulang), karena angka dari dua model yang berbeda tidak bisa dibandingkan.

---

## 4. Data dan cara menguji yang jujur

**Dataset:** Tongji Contactless Palmprint Dataset. 300 orang × 2 tangan = **600 telapak**, masing-masing 10 foto di **sesi 1** dan 10 foto di **sesi 2** (rata-rata 61 hari kemudian).

**Pembagian per telapak (bukan per foto)** di `training/data.py`, `buat_pembagian`:

| Kelompok | Porsi | Dipakai untuk |
|---|---|---|
| Latih | 70% (420 telapak) | Menyetel bobot model |
| Validasi | 10% (60 telapak) | Memilih epoch terbaik dan menentukan **batas** |
| Uji | 20% (120 telapak) | **Angka akhir di laporan**. Tidak pernah dilihat model saat training |

Kenapa per telapak? Kalau foto dari telapak yang sama ada di data latih *dan* data uji, model cukup "mengingat" telapak itu dan akurasinya akan terlihat tinggi palsu. Ini disebut **kebocoran data** (*data leakage*). Dengan pembagian per telapak, telapak uji berperan seperti orang baru yang mendaftar di aplikasi.

**Cara menguji** (`training/evaluasi.py`): untuk setiap telapak uji, 8 foto sesi 1 menjadi "pendaftaran" dan semua foto sesi 2 menjadi "scan". Ini meniru pemakaian sebenarnya: daftar hari ini, scan dua bulan kemudian.

**Augmentasi:** saat training, foto diubah sedikit secara acak (diputar ±10°, digeser, kecerahan dan kontras ±30%, sedikit blur dan noise), supaya model tahan terhadap perbedaan kamera dan lampu webcam.

---

## 5. Angka-angka yang dilaporkan

- **FAR** (*False Accept Rate*): berapa persen orang lain yang salah diterima.
- **FRR** (*False Reject Rate*): berapa persen pemilik yang salah ditolak.
- Menaikkan batas menurunkan FRR tetapi menaikkan FAR, dan sebaliknya.
- **EER** (*Equal Error Rate*): tingkat kesalahan pada batas di mana FAR = FRR. Ini satu angka untuk membandingkan sistem: **semakin kecil semakin baik**.
- **Rank-1**: berapa persen scan yang telapak terdekatnya memang telapak yang benar.
- **Kurva ROC**: menunjukkan semua pasangan FAR dan TAR (1 − FRR) untuk setiap batas.

### Hasil (model versi `20261122-2131`)

**Persiapan data** (`hasil/siapkan_data.txt`): dari 12.000 foto, **9.612 (80,1%)** berhasil dipotong menjadi ROI. Sisanya gagal karena MediaPipe tidak menemukan tangan (foto Tongji cukup gelap) atau karena lebih dari 10% kotak telapak keluar dari foto. Ada 426 telapak yang lengkap (≥8 ROI di sesi 1 dan ada ROI di sesi 2), dibagi menjadi 298 latih, 43 validasi, dan 85 uji. Ditambah 162 telapak yang tidak lengkap tetapi masih bisa dipakai untuk latih, totalnya **460 telapak latih (7.230 ROI)**.

**Training** (`hasil/grafik_training.png`, `hasil/riwayat_training.csv`): 20 epoch di GPU RTX 3050, sekitar 30 detik per epoch, total sekitar 10 menit.

| Epoch | Loss | Akurasi latih | EER validasi |
|---|---|---|---|
| 1 | 19,86 | 16,6% | 10,54% |
| 2 | 10,20 | 79,3% | 4,26% |
| **6 (dipilih)** | **0,42** | **99,8%** | **2,33%** |
| 20 | 0,01 | 100,0% | 2,78% |

Model yang disimpan adalah epoch 6, karena EER validasinya paling kecil. Setelah epoch 6, akurasi latih sudah 100% tetapi EER validasi tidak membaik lagi. Ini tanda awal **overfitting**: model mulai menghafal telapak latih alih-alih belajar ciri umum telapak. Karena itulah model dipilih berdasarkan data validasi, bukan data latih.

**Evaluasi akhir pada 85 telapak uji yang belum pernah dilihat** (`hasil/evaluasi.json`, `hasil/roc.png`, `hasil/histogram_jarak.png`). Pendaftaran memakai 8 foto sesi 1, scan memakai foto sesi 2 (sekitar 2 bulan kemudian):

| Ukuran | Nilai |
|---|---|
| EER | **3,98%** |
| Rank-1 (scan cocok dengan telapak yang benar di antara 85) | **86,6%** |
| Batas (dipilih dari validasi) | 0,619 |
| FAR pada batas | 3,57% |
| FRR pada batas | 4,26% |
| Jumlah pasangan diuji | 774 asli, 65.016 palsu |

Histogram jarak menunjukkan dua gunung yang hampir terpisah: pasangan telapak yang sama (hijau) berkumpul di jarak sekitar 0,2–0,5, sedangkan telapak yang berbeda (merah) di sekitar 0,7–1,0.

**Uji pada foto webcam saya sendiri** (`data/sendiri`, tangan kanan dan kiri, 8 foto daftar + 24 foto scan per tangan):

| Ukuran | Nilai |
|---|---|
| Rank-1 | **97,9%** (47 dari 48 scan cocok dengan tangan yang benar) |
| Jarak rata-rata tangan sama / tangan lain | 0,33 / 0,67 |
| EER | 10,4% (pada batas 0,51) |
| Pada batas 0,50 | FAR 6%, FRR 12% |
| Pada batas model 0,62 | FAR 35%, FRR 4% |

Hasil di webcam lebih buruk daripada di Tongji. Penyebabnya, model hanya dilatih dengan foto Tongji (kamera industri, lampu tetap), sedangkan webcam memiliki kamera, lampu, dan resolusi yang berbeda. Fenomena ini disebut *domain shift*. Selain itu, tangan kiri dan kanan orang yang sama lebih mirip satu sama lain daripada telapak orang yang berbeda. Untuk demo di webcam, batas di halaman Scan sebaiknya digeser ke sekitar **0,50**.

---

## 6. Keterbatasan (jujur)

- Dataset Tongji diambil dengan kamera industri di dalam kotak berlampu tetap, sedangkan aplikasi memakai webcam dengan lampu ruangan. Akurasi di webcam bisa lebih rendah daripada angka di data uji Tongji.
- Hanya satu dataset. Rencana berikutnya: mengumpulkan foto teman-teman sendiri (`backend/tools/ambil_foto.py`) untuk fine-tuning tambahan.
- Belum ada anti-pemalsuan (misalnya foto telapak di layar ponsel).

---

## 7. Pertanyaan yang mungkin diajukan dosen

**Apa yang Anda latih sendiri?**
Model ResNet18 yang mengubah gambar telapak menjadi 128 angka. Saya melatihnya (fine-tuning) dengan 420 telapak dataset Tongji memakai loss ArcFace. MediaPipe tidak saya latih, karena hanya dipakai untuk menemukan tangan.

**Kenapa tidak dilatih dari nol?**
Data telapak terbatas. Model dari nol butuh jauh lebih banyak data dan waktu. Pretrained ImageNet sudah mengerti garis dan tekstur, jadi tinggal diajari membedakan telapak.

**Kalau ada orang baru, apakah harus training ulang?**
Tidak. Model menghasilkan embedding, bukan menebak nama. Orang baru cukup mendaftar dengan 8 foto. Telapak di data uji pun tidak pernah dilihat model, dan itulah yang diukur oleh EER.

**Apa bedanya dengan klasifikasi biasa?**
Klasifikasi biasa menebak "ini kelas nomor berapa" dari daftar yang tetap. Di sini kepala klasifikasi (ArcFace) hanya alat bantu saat training dan dibuang sesudahnya. Yang dipakai adalah ruang embedding-nya.

**Kenapa memakai jarak kosinus?**
Embedding sudah dinormalisasi (panjangnya 1), sehingga yang membedakan hanya arahnya. Jarak kosinus mengukur seberapa beda arah dua vektor. ArcFace juga melatih model berdasarkan sudut, jadi keduanya cocok.

**Bagaimana batas ditentukan?**
Dari data validasi, pada titik EER. Lalu batas itu dipakai apa adanya pada data uji, supaya angka uji tidak "disetel" ke data uji.

**Bagaimana kalau fotonya buram atau tangannya mengepal?**
Ada validasi sebelum model: jari harus terbuka, telapak harus menghadap kamera, dan ROI tidak boleh terlalu gelap, terlalu terang, atau buram (`backend/validation.py`). Foto yang gagal ditolak dengan pesan.

**Di mana pengolahan citranya?**
Rotasi dan pelurusan telapak, pemotongan ROI, konversi ke abu-abu, resize, CLAHE, normalisasi, augmentasi, dan konvolusi di dalam CNN itu sendiri.

**Data apa yang disimpan? Apakah aman?**
Hanya 128 angka × 8 per tangan, ditambah nama, umur, dan status. Foto tidak disimpan. Dari angka-angka itu foto telapak tidak bisa dibuat ulang dengan mudah.

---

## 8. Cara menjalankan (ringkas)

```
.venv\Scripts\python -m training.data --asli data/tongji/asli --keluar data/roi
.venv\Scripts\python -m training.latih --roi data/roi --epoch 20
.venv\Scripts\python -m training.evaluasi --model model/telapak.pt --roi data/roi --sendiri data/sendiri
cd backend  &&  ..\.venv\Scripts\python app.py        (MySQL Laragon harus menyala)
cd frontend &&  npm run dev
```

---

## Sitasi dataset

L. Zhang, L. Li, A. Yang, Y. Shen, M. Yang, "Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach," *Pattern Recognition*, vol. 69, pp. 199–212, 2017.
