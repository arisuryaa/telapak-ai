"""latih.py - fine-tune ResNet18 with ArcFace on the training palms.

Every epoch the model sees every training ROI once (with small random changes, "augmentation"), the
ArcFace loss says how wrong its guesses were, and AdamW nudges all weights to be a little less wrong.
After each epoch the EER on the validation palms is measured; the epoch with the best EER is saved to
model/telapak.pt together with the threshold at that EER.

Run from the repo root (after python -m training.data):

    python -m training.latih --roi data/roi --epoch 20
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import training  # noqa: F401  (puts backend/ on sys.path)
from model import ArcFace, JaringanTelapak, Pengenal, siapkan_masukan, simpan_model
from training.data import SESI, roi_per_telapak
from training.evaluasi import evaluasi

# Small random changes so the model does not depend on one camera, light or exact crop.
AUGMENTASI = v2.Compose([
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    v2.ColorJitter(brightness=0.3, contrast=0.3),
    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
])
PELUANG_NOISE, NOISE = 0.3, 0.03 * 255


class DatasetTelapak(Dataset):
    """Every ROI (both sessions) of the palms `ids`. Label = position of the palm in `ids` (0..k-1)."""

    def __init__(self, folder_roi, ids, augmentasi):
        self.augmentasi = augmentasi
        posisi = {t: i for i, t in enumerate(ids)}
        self.roi, self.label = [], []
        for sesi in SESI:
            for t, paths in roi_per_telapak(folder_roi, sesi).items():
                if t in posisi:
                    for p in paths:  # all ROIs are kept in memory: about 16 KB each
                        self.roi.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
                        self.label.append(posisi[t])

    def __len__(self):
        return len(self.roi)

    def __getitem__(self, i):
        roi = self.roi[i]
        if self.augmentasi:
            roi = AUGMENTASI(torch.from_numpy(roi)[None])[0].numpy()
            if np.random.rand() < PELUANG_NOISE:
                roi = np.clip(roi + np.random.normal(0, NOISE, roi.shape), 0, 255).astype(np.uint8)
        return siapkan_masukan([roi])[0], self.label[i]


def _gambar_riwayat(riwayat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in riwayat]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(ep, [r["loss"] for r in riwayat], marker="o", color="#2563eb")
    ax[0].set_title("Loss latih (ArcFace)")
    ax[1].plot(ep, [100 * r["akurasi_latih"] for r in riwayat], marker="o", color="#16a34a")
    ax[1].set_title("Akurasi latih (%)")
    ax[2].plot(ep, [100 * r["eer_validasi"] for r in riwayat], marker="o", color="#dc2626")
    ax[2].set_title("EER validasi (%), makin kecil makin baik")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _simpan_riwayat(riwayat):
    Path("hasil").mkdir(exist_ok=True)
    with open("hasil/riwayat_training.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss", "akurasi_latih", "eer_validasi", "detik"])
        w.writeheader()
        w.writerows(riwayat)
    _gambar_riwayat(riwayat, "hasil/grafik_training.png")


def latih(folder_roi, pembagian, keluar, epoch=20, batch=64, lr=5e-4, pretrained=True, perangkat=None):
    """Train, save the epoch with the best validation EER to `keluar`, and return the history per epoch."""
    torch.manual_seed(42)
    np.random.seed(42)
    perangkat = perangkat or ("cuda" if torch.cuda.is_available() else "cpu")
    pakai_amp = perangkat == "cuda"
    versi = datetime.now().strftime("%Y%m%d-%H%M")

    data = DatasetTelapak(folder_roi, pembagian["latih"], augmentasi=True)
    loader = DataLoader(data, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    jaringan = JaringanTelapak(pretrained=pretrained).to(perangkat)
    kepala = ArcFace(len(pembagian["latih"])).to(perangkat)
    optimizer = torch.optim.AdamW(list(jaringan.parameters()) + list(kepala.parameters()), lr=lr, weight_decay=5e-4)
    jadwal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch * len(loader))
    skala = torch.amp.GradScaler("cuda", enabled=pakai_amp)
    print(f"Training {versi} di {perangkat}: {len(data)} ROI dari {len(pembagian['latih'])} telapak, "
          f"{len(loader)} batch per epoch, {epoch} epoch", flush=True)

    riwayat, terbaik = [], None
    for ep in range(1, epoch + 1):
        mulai = time.time()
        jaringan.train()
        kepala.train()
        total_loss, benar, jumlah = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(perangkat), y.to(perangkat)
            with torch.autocast("cuda", dtype=torch.float16, enabled=pakai_amp):
                e = jaringan(x)
                loss = F.cross_entropy(kepala(e.float(), y), y)
            optimizer.zero_grad(set_to_none=True)
            skala.scale(loss).backward()
            skala.step(optimizer)
            skala.update()
            jadwal.step()
            total_loss += float(loss) * len(y)
            with torch.no_grad():  # accuracy without the margin: which training palm is closest
                tebakan = F.linear(e.float(), F.normalize(kepala.bobot, dim=1)).argmax(dim=1)
            benar += int((tebakan == y).sum())
            jumlah += len(y)

        pengenal = Pengenal(jaringan, versi, None, perangkat)  # puts the network in eval mode
        val = evaluasi(pengenal, folder_roi, pembagian["validasi"])
        catatan = {"epoch": ep, "loss": round(total_loss / max(jumlah, 1), 4),
                   "akurasi_latih": round(benar / max(jumlah, 1), 4), "eer_validasi": round(val["eer"], 4),
                   "detik": round(time.time() - mulai, 1)}
        riwayat.append(catatan)
        lebih_baik = terbaik is None or val["eer"] < terbaik
        if lebih_baik:
            terbaik = val["eer"]
            simpan_model(keluar, jaringan, versi=versi, batas=val["batas"],
                         info={"epoch": ep, "eer_validasi": val["eer"], "telapak_latih": len(pembagian["latih"]),
                               "roi_latih": len(data), "lr": lr, "batch": batch, "epoch_total": epoch})
        print(f"epoch {ep:2d}/{epoch}  loss {catatan['loss']:.3f}  akurasi latih {100 * catatan['akurasi_latih']:5.1f}%  "
              f"EER validasi {100 * catatan['eer_validasi']:5.2f}%  ({catatan['detik']} s)"
              + ("  <- disimpan" if lebih_baik else ""), flush=True)
        _simpan_riwayat(riwayat)  # after every epoch, so the graph exists even if training is stopped early
    return riwayat


def main(argv=None):
    ap = argparse.ArgumentParser(description="Latih ResNet18 + ArcFace pada ROI telapak.")
    ap.add_argument("--roi", default="data/roi")
    ap.add_argument("--pembagian", default="data/pembagian.json")
    ap.add_argument("--keluar", default="model/telapak.pt")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args(argv)
    try:
        pembagian = json.loads(Path(args.pembagian).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Galat: {args.pembagian} belum ada. Jalankan dulu: python -m training.data", file=sys.stderr)
        return 1
    riwayat = latih(args.roi, pembagian, Path(args.keluar), epoch=args.epoch, batch=args.batch, lr=args.lr)
    terbaik = min(riwayat, key=lambda r: r["eer_validasi"])
    print(f"Selesai. Epoch terbaik {terbaik['epoch']} dengan EER validasi {100 * terbaik['eer_validasi']:.2f}%. "
          f"Model: {args.keluar}. Grafik: hasil/grafik_training.png")
    print("Berikutnya: python -m training.evaluasi --sendiri data/sendiri")
    return 0


if __name__ == "__main__":
    sys.exit(main())
