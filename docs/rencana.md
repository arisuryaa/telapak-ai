# Telapak AI: Rencana Implementasi

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ganti pengenal Canny/Gabor di Telapak ID dengan model CNN (ResNet18 + ArcFace) yang dilatih pada dataset Tongji, dengan alur daftar/scan yang tetap sama.

**Architecture:** `telapak-ai` adalah hasil `git clone` dari `telapak-id`. Model didefinisikan sekali di `backend/model.py` dan dipakai oleh training (`training/`) maupun backend Flask. ROI dataset dan ROI webcam dipotong dengan `palm.ambil_roi`. Database menyimpan 8 embedding float32 × 128 per tangan beserta versi model. Pencocokan memakai jarak kosinus terhadap sampel terdekat.

**Tech Stack:** Python 3.10 (`C:\laragon\bin\python\python-3.10\python.exe`), venv `.venv`, PyTorch + torchvision (CUDA, RTX 3050 6 GB), MediaPipe 0.10.14, numpy<2, Flask, SQLAlchemy + MySQL (Laragon), matplotlib, React/Vite.

**Spec:** `docs/rancangan.md`. Bagian 9 berlaku di atas bagian lain.

## Global Constraints

- Tenggat: 23 November 2026 malam. Kalau waktu mepet, ikuti urutan tugas dan jangan menambah fitur.
- `mediapipe==0.10.14`, `numpy<2` tetap dikunci.
- Model CNN satu-satunya metode. Tidak ada `METODE`, Canny, Gabor, atau `ROI_SKALA` di aplikasi; kotak ROI tetap 1.0.
- Database baru `telapak_ai` (`DATABASE_URL=mysql+pymysql://root:@localhost:3306/telapak_ai`), supaya database proyek lama `telapak_id` tidak tersentuh.
- `data/`, `model/`, `.venv/` masuk `.gitignore`. Dataset dan foto pemilik tidak pernah di-commit.
- Pembagian per telapak: 70% latih / 10% validasi / 20% uji, seed 42, disimpan di `data/pembagian.json`. Satu telapak hanya ada di satu kelompok.
- Embedding 128 angka dengan panjang 1. Jarak = 1 − kosinus. Batas dipilih di data **validasi** (titik EER), lalu dilaporkan di data **uji**.
- Teks yang dilihat pengguna memakai bahasa Indonesia. Nama fungsi mengikuti gaya proyek (bahasa Indonesia).
- Commit di akhir setiap tugas, tanpa baris Co-Authored-By (permintaan pemilik).

## Review Focus

1. Foto Tongji yang gagal dideteksi MediaPipe harus **dilewati dan dihitung**, bukan menghentikan seluruh proses. Laporannya menyebut persentase berhasil. (Tes ada di Tugas 2.)
2. Kalau backend dijalankan tanpa file `model/telapak.pt`, harus muncul pesan yang jelas ("jalankan training dulu"), bukan traceback. (Tes ada di Tugas 6.)
3. Tangan yang terdaftar dengan versi model lama tidak boleh dibandingkan dengan model baru. Scan menyarankan pendaftaran ulang. (Tes ada di Tugas 6.)
4. Sampel dari browser yang rusak (ukuran bukan 512 byte, atau berisi NaN/inf) harus ditolak dengan "Data sampel tidak lengkap". (Tes ada di Tugas 6.)
5. Telapak yang punya kurang dari 9 foto ROI di sesi 1, atau tidak punya foto di sesi 2, dilewati saat evaluasi, tanpa membuat pembagian atau EER error. (Tes ada di Tugas 4.)

---

### Tugas 1: Clone, lingkungan Python, dan PyTorch

**Files:**
- Create: `C:\Users\Dewa\Desktop\telapak-ai\` (hasil clone), `.venv/`, `data/sendiri/`
- Modify: `.gitignore`, `backend/requirements.txt`, `backend/.env.example`

**Interfaces:**
- Produces: `.venv\Scripts\python.exe` dengan torch, torchvision, dan matplotlib; folder `data/`, `model/`, `hasil/`.

- [ ] **Step 1:** Pindahkan `telapak-ai\docs` sementara ke scratchpad, jalankan `git clone C:\Users\Dewa\Desktop\telapak-id C:\Users\Dewa\Desktop\telapak-ai`, kembalikan `docs\`, lalu `git remote remove origin` supaya repo ini tidak terhubung ke proyek lama.
- [ ] **Step 2:** Buat venv: `C:\laragon\bin\python\python-3.10\python.exe -m venv .venv`. Pasang `pip install -r backend/requirements.txt`, lalu `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` dan `pip install matplotlib`. Tambahkan `torch`, `torchvision`, `matplotlib` ke `requirements.txt` (dengan komentar soal index CUDA).
- [ ] **Step 3:** Verifikasi: `.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. Hasil yang diharapkan: `True` dan `RTX 3050`. Kalau gagal lebih dari 30 menit, catat dan lanjutkan dengan CPU untuk tes; training pindah ke Colab (rancangan 9.9).
- [ ] **Step 4:** `.gitignore`: tambahkan `data/`, `model/`, `.venv/`. Salin `telapak-id\backend\data_uji\*` ke `data\sendiri\`. Di `.env.example` dan `.env`, ganti database menjadi `telapak_ai` dan hapus `ROI_SKALA` serta `METODE`.
- [ ] **Step 5:** Jalankan `cd backend; ..\.venv\Scripts\python -m pytest -q`. Hasil yang diharapkan: semua tes lama lolos (clone sehat). Commit: `chore: clone telapak-id sebagai telapak-ai dan siapkan lingkungan training`.

### Tugas 2: Menyiapkan data Tongji menjadi ROI 128×128

**Files:**
- Create: `training/__init__.py`, `training/data.py`, `training/tests/conftest.py`, `training/tests/test_data.py`

**Interfaces:**
- Consumes: `palm.deteksi_tangan(frame) -> (lm, label) | None`, `palm.ambil_roi(frame, lm) -> uint8[128,128] | None`
- Produces:
  - `id_telapak(nama_file: str) -> int`: `"00001.bmp"` → 0, `"00010"` → 0, `"00011"` → 1 (aturan Tongji: setiap 10 nomor = 1 telapak).
  - `siapkan_roi(folder_asli: Path, folder_keluar: Path, deteksi=palm.deteksi_tangan) -> dict`: membaca `session1/` dan `session2/` (semua ekstensi gambar), menulis `folder_keluar/sessionN/<stem>.png`, dan mengembalikan `{"total": int, "berhasil": int, "gagal": [nama, ...]}`.
  - `salin_roi_bawaan(folder_roi: Path, folder_keluar: Path) -> dict`: ROI bawaan Tongji → abu-abu 128×128 PNG dengan struktur yang sama.
  - `buat_pembagian(folder_roi: Path, seed=42) -> dict`: `{"latih": [id...], "validasi": [...], "uji": [...]}` dengan proporsi 70/10/20, hanya dari telapak yang punya ≥9 ROI di session1 dan ≥1 di session2.
  - CLI: `python -m training.data --asli data/tongji/asli --keluar data/roi [--bawaan data/tongji/roi]`, yang menulis `data/pembagian.json` dan `hasil/siapkan_data.txt` (persentase berhasil).
- `training/tests/conftest.py` menambahkan `backend/` ke `sys.path`, menyediakan `deteksi_tiruan` (landmark dari `backend/tests/fixtures/landmarks_test.json`, `terbuka_kanan`), dan `buat_foto_tongji(folder, sesi, nomor_list)` yang menulis frame 640×480 bertekstur acak.

- [ ] **Step 1: Tulis tes yang gagal**

```python
def test_id_telapak():
    assert [id_telapak(n) for n in ["00001.bmp", "00010", "00011.jpg", "06000"]] == [0, 0, 1, 599]

def test_siapkan_roi_melewati_foto_gagal(tmp_path, deteksi_tiruan, buat_foto_tongji):
    buat_foto_tongji(tmp_path / "asli", "session1", range(1, 4))
    (tmp_path / "asli" / "session1" / "00004.bmp").write_bytes(b"rusak")
    hasil = siapkan_roi(tmp_path / "asli", tmp_path / "roi", deteksi=deteksi_tiruan)
    assert hasil["total"] == 4 and hasil["berhasil"] == 3 and hasil["gagal"] == ["session1/00004.bmp"]
    roi = cv2.imread(str(tmp_path / "roi" / "session1" / "00001.png"), cv2.IMREAD_GRAYSCALE)
    assert roi.shape == (128, 128)

def test_pembagian_per_telapak_tidak_bocor(tmp_path):
    # 20 telapak lengkap + 1 telapak dengan hanya 5 foto di session1 (harus dilewati)
    ...buat PNG kosong 128x128 untuk session1 (10/telapak) dan session2 (10/telapak)...
    p = buat_pembagian(tmp_path)
    semua = p["latih"] + p["validasi"] + p["uji"]
    assert len(semua) == len(set(semua)) == 20 and 20 not in semua
    assert (len(p["latih"]), len(p["validasi"]), len(p["uji"])) == (14, 2, 4)
    assert buat_pembagian(tmp_path) == p  # seed tetap = hasil sama
```

- [ ] **Step 2:** `.venv\Scripts\python -m pytest training/tests/test_data.py -v`. Hasil yang diharapkan: FAIL (modul belum ada).
- [ ] **Step 3:** Implementasikan `training/data.py`. Foto abu-abu diubah ke BGR sebelum dideteksi. Foto yang tidak terbaca, tangan yang tidak terdeteksi, atau ROI yang `None` masuk ke `gagal`. Pembulatan proporsi: `uji = round(0.2n)`, `validasi = round(0.1n)`, sisanya latih.
- [ ] **Step 4:** Jalankan tes. Hasil yang diharapkan: PASS.
- [ ] **Step 5:** Commit: `feat: siapkan ROI dataset Tongji dan pembagian per telapak`.

### Tugas 3: Model ResNet18 + ArcFace

**Files:**
- Create: `backend/model.py`, `backend/tests/test_model.py`

**Interfaces:**
- Consumes: `palm.terapkan_clahe(roi) -> uint8[128,128]` (tetap di `palm.py`)
- Produces:
  - `DIMENSI = 128`
  - `siapkan_masukan(roi_list: list[np.ndarray uint8 128×128]) -> torch.Tensor [N,3,128,128]`: CLAHE, lalu /255, salin ke 3 kanal, lalu normalisasi ImageNet (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`).
  - `class JaringanTelapak(nn.Module)`: `__init__(pretrained: bool = True)` memakai `torchvision.models.resnet18(weights=IMAGENET1K_V1 or None)`, lalu `fc` diganti dengan `Linear(512, 128)` + `BatchNorm1d(128)`. `forward(x) -> [N,128]` dinormalisasi L2.
  - `class ArcFace(nn.Module)`: `__init__(jumlah_kelas: int, s=30.0, m=0.5)`, `forward(embedding, label) -> logits [N, jumlah_kelas]` (additive angular margin).
  - `simpan_model(path, jaringan, versi: str, batas: float | None, info: dict)` dan `class Pengenal` dengan konstruktor `Pengenal(jaringan, versi: str, batas: float | None, perangkat)` (dipakai `latih.py` untuk evaluasi validasi di tengah training) dan `Pengenal.dari_file(path, perangkat=None) -> Pengenal` (memilih CUDA kalau tersedia), atribut `.versi: str`, `.batas: float | None`, `.embedding(roi) -> np.float32[128]`, `.embedding_banyak(roi_list) -> np.float32[N,128]`, `.detail(roi) -> {"masukan": uint8[128,128] (setelah CLAHE), "fitur": list[8 × uint8 32×32] (8 kanal pertama layer1, diskala 0–255), "embedding": float32[128]}`.
  - Kalau path tidak ada, `dari_file` melempar `FileNotFoundError("Model belum ada di <path>. Jalankan training dulu: python -m training.latih")`.

- [ ] **Step 1: Tulis tes yang gagal** (memakai `pretrained=False` supaya tidak mengunduh bobot):

```python
def test_embedding_panjang_satu():
    j = JaringanTelapak(pretrained=False).eval()
    e = j(siapkan_masukan([np.full((128, 128), v, np.uint8) for v in (30, 200)]))
    assert e.shape == (2, 128) and torch.allclose(e.norm(dim=1), torch.ones(2), atol=1e-5)

def test_arcface_bentuk_dan_margin():
    a = ArcFace(5)
    e = F.normalize(torch.randn(4, 128), dim=1)
    lab = torch.tensor([0, 1, 2, 3])
    keluar = a(e, lab)
    assert keluar.shape == (4, 5)
    tanpa_margin = 30.0 * F.linear(e, F.normalize(a.bobot, dim=1))
    assert (keluar.gather(1, lab[:, None]) < tanpa_margin.gather(1, lab[:, None])).all()

def test_simpan_muat_pengenal(tmp_path):
    j = JaringanTelapak(pretrained=False)
    simpan_model(tmp_path / "m.pt", j, versi="uji-1", batas=0.4, info={"epoch": 1})
    p = Pengenal.dari_file(tmp_path / "m.pt", perangkat="cpu")
    roi = np.random.default_rng(0).integers(0, 256, (128, 128), dtype=np.uint8)
    assert p.versi == "uji-1" and p.batas == 0.4
    assert p.embedding(roi).dtype == np.float32 and p.embedding(roi).shape == (128,)
    np.testing.assert_allclose(p.embedding(roi), p.embedding_banyak([roi])[0], atol=1e-5)
    d = p.detail(roi)
    assert d["masukan"].shape == (128, 128) and len(d["fitur"]) == 8 and d["fitur"][0].dtype == np.uint8

def test_model_tidak_ada_pesan_jelas(tmp_path):
    with pytest.raises(FileNotFoundError, match="Jalankan training dulu"):
        Pengenal.dari_file(tmp_path / "tidak_ada.pt")
```

- [ ] **Step 2:** Jalankan tes. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan `backend/model.py`. Nama parameter ArcFace adalah `bobot` ([jumlah_kelas, 128]). ArcFace: `cos = linear(e, normalize(bobot))`, lalu `cos(θ+m)` untuk kelas yang benar, dengan versi "easy margin" bila `θ+m > π`. Checkpoint menyimpan `{"bobot": state_dict, "dimensi": 128, "versi", "batas", "info"}`. `detail()` memakai forward hook pada `layer1`.
- [ ] **Step 4:** Jalankan tes. Hasil yang diharapkan: PASS.
- [ ] **Step 5:** Commit: `feat: model ResNet18 dengan embedding 128 dan kepala ArcFace`.

### Tugas 4: Metrik dan evaluasi

**Files:**
- Create: `training/metrik.py`, `training/evaluasi.py`, `training/tests/test_metrik.py`

**Interfaces:**
- Produces:
  - `jarak_kosinus(a: np.ndarray [N,128], b: np.ndarray [M,128]) -> np.ndarray [N,M]` (1 − a·bᵀ).
  - `skor_identifikasi(daftar: dict[int, np.ndarray [k,128]], scan: list[tuple[int, np.ndarray 128]]) -> (asli: np.ndarray, palsu: np.ndarray, rank1: float)`: jarak scan ke sebuah telapak = jarak minimum ke sampelnya. `asli` = jarak ke telapaknya sendiri, `palsu` = jarak ke semua telapak lain.
  - `eer(asli, palsu) -> (eer: float, batas: float)`: menyapu batas di antara semua nilai, lalu mengambil titik di mana FAR ≈ FRR.
  - `roc(asli, palsu) -> (far: np.ndarray, tar: np.ndarray)`.
  - `evaluasi(pengenal, folder_roi, ids: list[int], jumlah_daftar=8) -> dict` dengan kunci `eer, batas, rank1, jumlah_telapak, jumlah_scan, asli, palsu`. Daftar = 8 ROI pertama session1 per telapak (urut nama), scan = semua ROI session2. Telapak yang tidak lengkap dilewati.
  - CLI `python -m training.evaluasi --model model/telapak.pt --roi data/roi`: batas diambil dari **validasi** lalu disimpan ke file model (`simpan_model` dengan `batas` baru). Skor dihitung pada **uji**. Hasilnya ditulis ke `hasil/evaluasi.json`, `hasil/roc.png`, `hasil/histogram_jarak.png`, dan ringkasan dicetak ke layar (EER, rank-1, batas). Opsi `--sendiri data/sendiri` mengevaluasi foto pemilik (folder per tangan, daftar 8 pertama, sisanya scan) memakai `palm.deteksi_tangan` + `ambil_roi`.

- [ ] **Step 1: Tulis tes yang gagal**

```python
def test_eer_terpisah_sempurna_dan_tumpang_tindih():
    e, b = eer(np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.7, 0.8]))
    assert e == 0.0 and 0.3 <= b < 0.6
    e2, _ = eer(np.array([0.1, 0.5]), np.array([0.4, 0.9]))
    assert e2 == pytest.approx(0.5)

def test_skor_identifikasi_sampel_terdekat():
    v = np.eye(128, dtype=np.float32)
    daftar = {0: v[[0, 1]], 1: v[[2, 3]]}
    asli, palsu, rank1 = skor_identifikasi(daftar, [(0, v[1]), (1, v[0])])
    assert asli.tolist() == [0.0, 1.0] and sorted(palsu.tolist()) == [0.0, 1.0] and rank1 == 0.5

def test_evaluasi_melewati_telapak_tidak_lengkap(tmp_path):
    ...ROI PNG acak: telapak 0 dan 1 lengkap, telapak 2 hanya punya 5 foto session1...
    hasil = evaluasi(PengenalPalsu(), tmp_path, [0, 1, 2])
    assert hasil["jumlah_telapak"] == 2 and 0.0 <= hasil["eer"] <= 1.0
```

`PengenalPalsu` (di `training/tests/conftest.py`): `embedding_banyak` = ROI diperkecil menjadi 8×16, diratakan, lalu dinormalisasi. Hasilnya deterministik tanpa torch.

- [ ] **Step 2:** Jalankan tes. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan `metrik.py` dan `evaluasi.py`. Grafik dibuat dengan matplotlib (backend `Agg`). Label sumbu memakai bahasa Indonesia ("Jarak", "Jumlah pasangan", "Sama", "Beda").
- [ ] **Step 4:** Jalankan tes. Hasil yang diharapkan: PASS.
- [ ] **Step 5:** Commit: `feat: metrik EER, rank-1, ROC dan skrip evaluasi`.

### Tugas 5: Training

**Files:**
- Create: `training/latih.py`, `training/tests/test_latih.py`

**Interfaces:**
- Consumes: `JaringanTelapak`, `ArcFace`, `siapkan_masukan`, `simpan_model`, `Pengenal` (Tugas 3); `evaluasi` dan `eer` (Tugas 4); `data/pembagian.json` (Tugas 2).
- Produces:
  - `class DatasetTelapak(torch.utils.data.Dataset)`: `__init__(folder_roi, ids, augmentasi: bool)`. Isinya semua ROI session1+session2 dari `ids`, dengan label = indeks urut id (0..k-1). `__getitem__` mengembalikan `(tensor [3,128,128], label)` memakai `siapkan_masukan`. Augmentasi (hanya untuk latih): `RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))`, kecerahan/kontras ±30%, `GaussianBlur` dengan p=0.3, dan noise Gaussian σ=0.03 dengan p=0.3.
  - `latih(folder_roi, pembagian, keluar: Path, epoch=20, batch=64, lr=5e-4, pretrained=True, perangkat=None) -> list[dict]`: AdamW (weight decay 5e-4), cosine schedule, AMP saat CUDA. Setiap epoch mencatat `{"epoch", "loss", "akurasi_latih", "eer_validasi"}`. Model dengan `eer_validasi` terbaik disimpan ke `keluar` (versi = `datetime.now().strftime("%Y%m%d-%H%M")`, batas = batas EER validasi). Riwayat ditulis ke `hasil/riwayat_training.csv` dan `hasil/grafik_training.png` (loss, akurasi, EER validasi per epoch).
  - CLI: `python -m training.latih --roi data/roi --epoch 20`.

- [ ] **Step 1: Tulis tes yang gagal** (CPU, data kecil):

```python
def test_latih_singkat_menyimpan_model_dan_riwayat(tmp_path, monkeypatch):
    ...ROI PNG acak untuk 6 telapak × (10 session1 + 2 session2)...
    pembagian = {"latih": [0, 1, 2, 3], "validasi": [4, 5], "uji": []}
    monkeypatch.chdir(tmp_path)
    riwayat = latih(tmp_path / "roi", pembagian, tmp_path / "m.pt", epoch=2, batch=8, pretrained=False, perangkat="cpu")
    assert [r["epoch"] for r in riwayat] == [1, 2] and all(np.isfinite(r["loss"]) for r in riwayat)
    p = Pengenal.dari_file(tmp_path / "m.pt", perangkat="cpu")
    assert p.batas is not None and p.versi
    assert (tmp_path / "hasil" / "grafik_training.png").exists()

def test_dataset_label_berurutan(tmp_path):
    ...ROI untuk id 7 dan 3...
    ds = DatasetTelapak(tmp_path, [7, 3], augmentasi=False)
    x, y = ds[0]
    assert x.shape == (3, 128, 128) and set(ds.label) == {0, 1}
```

- [ ] **Step 2:** Jalankan tes. Hasil yang diharapkan: FAIL.
- [ ] **Step 3:** Implementasikan `training/latih.py`. Folder `hasil/` dibuat relatif terhadap folder kerja. `num_workers=0` (paling aman di Windows).
- [ ] **Step 4:** Jalankan tes. Hasil yang diharapkan: PASS. Lalu jalankan semua tes `training/tests`.
- [ ] **Step 5:** Commit: `feat: skrip training ResNet18 + ArcFace dengan pencatatan per epoch`.

### Tugas 6: Backend memakai model (Canny/Gabor dihapus)

**Files:**
- Delete: `backend/canny.py`, `backend/metode.py`, `backend/tools/bandingkan_metode.py`, `backend/tools/bandingkan_skala.py`, `backend/tests/test_canny.py`, `backend/tests/test_bandingkan.py`, `backend/tests/fixtures/code_golden.npy`
- Modify: `backend/palm.py` (hapus `GABOR`, `_buat_filter`, `_FILTER`, `hitung_kode*`, `jarak_kode`, `jarak_ke_orang`, `versi_bentuk`; pertahankan deteksi, `batas_vertikal`, `ambil_roi*`, `terapkan_clahe`), `backend/pipeline.py`, `backend/steps.py`, `backend/db.py`, `backend/app.py`, `backend/tests/test_app.py`, `test_pipeline.py`, `test_db.py`, `test_palm.py`

**Interfaces:**
- Consumes: `Pengenal` (Tugas 3).
- Produces (dipakai frontend):
  - `analisis(frame, pengenal, deteksi=palm.deteksi_tangan) -> Analisis`. `Analisis.kode_detail = pengenal.detail(roi)`. `Analisis.kode` = embedding float32[128]. Atribut `roi_skala` dan `metode` dihapus.
  - `db.Telapak`: kolom `kode` (8×128 float32 = 4096 byte) dan `model_versi: String(20)`. Kolom `roi_skala`, `roi_bentuk`, `metode`, serta `_migrasi` dihapus (database baru). `kode_ke_bytes`/`bytes_ke_kode` memakai float32 dengan bentuk `(-1, 128)`. `simpan_telapak(s, orang_id, sisi, kode, ganti=False, model_versi="")`. `muat_galeri` menyertakan `model_versi`. `orang_ke_dict` mengganti `skala_tangan`/`metode_tangan` dengan `versi_tangan: {sisi: model_versi}`.
  - `create_app(database_url=None, deteksi=palm.deteksi_tangan, pengenal=None)`: kalau `pengenal` None, muat `Pengenal.dari_file(<repo>/model/telapak.pt)`.
  - `GET /api/pengaturan` → `{"metode": "cnn", "metode_nama": "Model CNN (ResNet18)", "model_versi", "batas_default": pengenal.batas or 0.4, "batas_min": 0.05, "batas_maks": 1.0, "roi_skala": 1.0}`.
  - `/api/scan` mengembalikan `skala_maks: 1.0`. Galeri difilter `model_versi == pengenal.versi`. Kalau ada tangan dengan versi lain tapi tidak ada yang cocok, pesannya: `"Belum ada yang terdaftar dengan model ini (versi <v>). Daftar ulang dulu."`. Log `dibandingkan: <nama> (<sisi>) jarak=…, per sampel: …` dipertahankan.
  - `/api/sampel` mengembalikan `kode` = base64 dari 512 byte float32. `_decode_sampel` menolak panjang ≠ 512 atau nilai yang tidak finite.
  - Log saat menyimpan: `tersimpan: orang_id=… sisi=… ganti=… model=<versi>, 4096 byte = 8 sampel x 128 angka`, lalu per sampel `  isi sampel i: 5 angka pertama …, panjang 1.000`.
  - `steps.py`: langkah 1–4 tidak berubah (tanpa baris "ukuran kotak"). Langkah 5 "Masukan model" (gambar `masukan`, penjelasan CLAHE dan normalisasi). Langkah 6 "Model CNN: peta fitur" (8 gambar `fitur`, penjelasan bahwa ini yang "dilihat" lapisan awal ResNet18). Langkah 7 "Embedding 128 angka" (gambar 8×16 kotak berwarna dari nilai embedding, diperbesar ×16; angka: dimensi, versi model). Langkah 8 "Pencocokan" dengan penjelasan jarak kosinus.
  - Saat `__main__`, `FileNotFoundError` dari model dicetak sebagai pesan lalu `exit(1)`, sama seperti `DatabaseTidakTersedia`.
- Tes backend memakai `PengenalPalsu` di `backend/tests/conftest.py`, dengan antarmuka yang sama seperti `Pengenal` (`versi="palsu-1"`, `batas=0.4`, embedding = ROI diperkecil 8×16, diratakan, lalu dinormalisasi; `detail` mengembalikan `masukan`, `fitur` berupa 8 salinan 32×32, dan `embedding`).

- [ ] **Step 1: Tulis tes yang gagal / perbarui tes**: alur lama (`test_daftar_lalu_dikenali`, `test_kepal_tidak_pernah_dikenali`, `test_hapus`, `test_simpan_telapak_duplikat`, `test_api_mencatat_log`) disesuaikan ke `PengenalPalsu`. Tambahkan tes berikut:

```python
def test_sampel_berisi_512_byte_float32(client): ...assert len(base64.b64decode(s["kode"])) == 512
def test_sampel_rusak_ditolak(client):  # 511 byte, dan 512 byte berisi NaN
    ...assert r.status_code == 400 and r.get_json()["pesan"] == "Data sampel tidak lengkap"
def test_versi_model_lain_tidak_dibandingkan(tmp_path):
    # daftar dengan PengenalPalsu(versi="lama"), scan dengan PengenalPalsu(versi="baru")
    ...assert hasil["status"] == "tidak_dikenal" and "Daftar ulang" in hasil["pesan"] and jarak == []
def test_pengaturan_model(client):
    p = client.get("/api/pengaturan").get_json()
    assert (p["metode"], p["model_versi"], p["batas_default"], p["roi_skala"]) == ("cnn", "palsu-1", 0.4, 1.0)
def test_langkah_model(client): ...judul langkah 5–7 == ["Masukan model", "Model CNN: peta fitur", "Embedding 128 angka"]
def test_tanpa_file_model_pesan_jelas(tmp_path, monkeypatch): ...create_app tanpa pengenal & tanpa model → FileNotFoundError "Jalankan training dulu"
```

Tes yang khusus untuk metode/skala/Gabor dihapus (`test_pengaturan_dan_skala_berbeda`, `test_scan_abaikan_data_*`, `test_*metode*`, `test_kode_sama_dengan_versi_lama`, `test_kode_detail_konsisten`, `test_jarak_diri_nol_acak_setengah`, `test_batas_vertikal_dan_versi_bentuk` bagian versi, tes migrasi di `test_db.py`).
- [ ] **Step 2:** Jalankan `cd backend; ..\.venv\Scripts\python -m pytest -q`. Hasil yang diharapkan: tes baru FAIL.
- [ ] **Step 3:** Hapus file Canny/Gabor dan ubah modul sesuai Interfaces.
- [ ] **Step 4:** Jalankan semua tes backend dan training. Hasil yang diharapkan: semua PASS, dan `grep -ri "canny\|gabor\|metode_mod\|roi_skala" backend --include=*.py` hanya menemukan `palm.py` (parameter `skala` di `ambil_roi_detail`) atau tidak ada sama sekali.
- [ ] **Step 5:** Commit: `feat: backend memakai model CNN sebagai satu-satunya metode`.

### Tugas 7: Frontend disesuaikan

**Files:**
- Modify: `frontend/src/lib/format.js`, `frontend/src/pages/Scan.jsx`, dan teks yang menyebut Gabor/Canny di `frontend/src` (cari dengan grep)

**Interfaces:**
- Consumes: `/api/pengaturan` dan `/api/scan` dari Tugas 6.

- [ ] **Step 1:** `format.js`: `formatTangan(orang)` → `"kanan (model 20261123-0130), kiri (model …)"` dari `versi_tangan`. `Scan.jsx`: `CADANGAN = { metode: "cnn", metode_nama: "Model CNN (ResNet18)", batas_default: 0.4, batas_min: 0.05, batas_maks: 1.0 }`. Kunci localStorage menyertakan versi: `batas_${p.metode}_${p.model_versi}`, supaya batas lama tidak terbawa setelah training ulang.
- [ ] **Step 2:** `grep -rn "Gabor\|Canny\|gabor\|canny" frontend/src`. Ganti teks yang masih relevan ke model, lalu jalankan `npm run test` (validation.test.js) dan `npm run build`. Hasil yang diharapkan: lolos dan build sukses.
- [ ] **Step 3:** Commit: `feat: frontend menampilkan model CNN`.

### Tugas 8: Menjalankan training sungguhan (dipandu, bersama pemilik)

**Files:**
- Create (hasil, tidak di-commit): `data/tongji/asli/`, `data/tongji/roi/`, `data/roi/`, `data/pembagian.json`, `model/telapak.pt`, `hasil/*`

- [ ] **Step 1 (pemilik):** Ekstrak *Original Images* ke `data/tongji/asli/` (harus berisi `session1/` dan `session2/`) dan *ROI Images* ke `data/tongji/roi/`. Claude memeriksa struktur, ekstensi, dan apakah fotonya berwarna atau abu-abu.
- [ ] **Step 2:** `.venv\Scripts\python -m training.data --asli data/tongji/asli --keluar data/roi`. Kalau persentase berhasil < 80%, beri tahu pemilik lalu jalankan ulang dengan `--bawaan data/tongji/roi` (rancangan 9.4). Catat pilihannya di `hasil/siapkan_data.txt`.
- [ ] **Step 3:** `.venv\Scripts\python -m training.latih --roi data/roi --epoch 20` (sekitar 30–60 menit di GPU; jalankan di background). Hasil yang diharapkan: EER validasi turun dari epoch ke epoch.
- [ ] **Step 4:** `.venv\Scripts\python -m training.evaluasi --model model/telapak.pt --roi data/roi --sendiri data/sendiri`. Catat EER, rank-1, dan batas.
- [ ] **Step 5:** Demo bersama pemilik: nyalakan MySQL, jalankan backend dan frontend, daftarkan tangan pemilik, lalu scan tangan kanan dan kiri. Hasil yang diharapkan: tangan yang benar dikenali dan tangan yang lain tidak dikenali (atau jaraknya lebih besar).

### Tugas 9: Penjelasan untuk dosen dan README

**Files:**
- Create: `docs/penjelasan.md`
- Modify: `README.md`

- [ ] **Step 1:** `docs/penjelasan.md` dalam bahasa sederhana: alur sistem (diagram teks), apa itu model dan training, pretrained dan fine-tuning, ResNet18, embedding, ArcFace (kepala dibuang setelah training), alasan pembagian per telapak dan latih/validasi/uji, arti EER, rank-1, dan batas, **angka hasil nyata dari Tugas 8** dengan gambar dari `hasil/`, keterbatasan (beda kamera Tongji vs webcam, satu dataset), sitasi dataset Tongji, serta 8–10 pertanyaan yang mungkin diajukan dosen beserta jawabannya.
- [ ] **Step 2:** `README.md`: ganti uraian Canny/Gabor dengan model. Tambahkan cara menyiapkan data, melatih, mengevaluasi, dan menjalankan aplikasi. Catat bahwa `model/telapak.pt` dibuat dari training dan tidak ada di git.
- [ ] **Step 3:** Commit: `docs: penjelasan cara kerja model dan cara menjalankan`.
