"""model.py - the CNN that turns a 128x128 palm ROI into 128 numbers (an embedding).

Used by training (training/latih.py) and by the web app, so both run exactly the same code.

  JaringanTelapak  ResNet18 pretrained on ImageNet, its last layer replaced by one that outputs 128 numbers,
                   scaled to length 1. Same palm -> embeddings pointing the same way.
  ArcFace          only used while training: it makes the model guess which training palm a photo belongs to,
                   with an angular margin so photos of one palm cluster tightly. Thrown away afterwards, so a
                   new person can enrol without retraining.
  Pengenal         a trained model loaded for use: embedding(roi), detail(roi) for the results page.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import palm

DIMENSI = 128
RATA_IMAGENET = (0.485, 0.456, 0.406)
SIMPANGAN_IMAGENET = (0.229, 0.224, 0.225)


def siapkan_masukan(roi_list):
    """uint8 128x128 grey ROIs -> model input [N,3,128,128]: CLAHE, 0..1, three channels, ImageNet normalisation."""
    g = np.stack([palm.terapkan_clahe(r) for r in roi_list]).astype(np.float32) / 255.0
    x = torch.from_numpy(g)[:, None].repeat(1, 3, 1, 1)
    rata = torch.tensor(RATA_IMAGENET).view(1, 3, 1, 1)
    simpangan = torch.tensor(SIMPANGAN_IMAGENET).view(1, 3, 1, 1)
    return (x - rata) / simpangan


class JaringanTelapak(nn.Module):
    """ResNet18 whose 1000-class ImageNet layer is replaced by a 128-number embedding of length 1."""

    def __init__(self, pretrained=True):
        super().__init__()
        bobot = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = torchvision.models.resnet18(weights=bobot)
        self.resnet.fc = nn.Sequential(nn.Linear(512, DIMENSI), nn.BatchNorm1d(DIMENSI))

    def forward(self, x):
        return F.normalize(self.resnet(x), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin head (Deng et al., 2019). logits = s * cos(angle + m) for the true palm."""

    def __init__(self, jumlah_kelas, s=30.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.bobot = nn.Parameter(torch.empty(jumlah_kelas, DIMENSI))
        nn.init.xavier_uniform_(self.bobot)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.ambang = math.cos(math.pi - m)  # beyond this, angle + m would pass 180 degrees
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        cos = F.linear(F.normalize(embedding, dim=1), F.normalize(self.bobot, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos ** 2)
        dengan_margin = cos * self.cos_m - sin * self.sin_m  # cos(angle + m)
        dengan_margin = torch.where(cos > self.ambang, dengan_margin, cos - self.mm)
        benar = F.one_hot(label, num_classes=self.bobot.shape[0]).bool()
        return self.s * torch.where(benar, dengan_margin, cos)


def simpan_model(path, jaringan, versi, batas, info):
    """Save the trained network with its version, the suggested threshold and training info."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bobot": jaringan.state_dict(), "dimensi": DIMENSI, "versi": versi, "batas": batas, "info": info}, path)


def _ke_0_255(peta):
    peta = peta - peta.min()
    puncak = float(peta.max())
    return (peta / puncak * 255).astype(np.uint8) if puncak > 0 else np.zeros(peta.shape, np.uint8)


class Pengenal:
    """A model ready for use: ROI in, embedding out."""

    def __init__(self, jaringan, versi, batas, perangkat=None, info=None):
        self.perangkat = perangkat or ("cuda" if torch.cuda.is_available() else "cpu")
        self.jaringan = jaringan.to(self.perangkat).eval()
        self.versi, self.batas, self.info = versi, batas, info or {}

    @classmethod
    def dari_file(cls, path, perangkat=None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model belum ada di {path}. Jalankan training dulu: python -m training.latih")
        data = torch.load(path, map_location="cpu", weights_only=True)
        jaringan = JaringanTelapak(pretrained=False)
        jaringan.load_state_dict(data["bobot"])
        return cls(jaringan, data["versi"], data["batas"], perangkat, data.get("info"))

    @torch.no_grad()
    def embedding_banyak(self, roi_list):
        """[N,128] float32 embeddings of a list of uint8 128x128 ROIs."""
        x = siapkan_masukan(roi_list).to(self.perangkat)
        return self.jaringan(x).float().cpu().numpy()

    def embedding(self, roi):
        return self.embedding_banyak([roi])[0]

    @torch.no_grad()
    def detail(self, roi):
        """What the results page shows: the input after CLAHE, 8 feature maps of the first ResNet block, the embedding."""
        tangkap = {}
        kait = self.jaringan.resnet.layer1.register_forward_hook(lambda m, i, o: tangkap.setdefault("fitur", o))
        try:
            e = self.jaringan(siapkan_masukan([roi]).to(self.perangkat)).float().cpu().numpy()[0]
        finally:
            kait.remove()
        fitur = tangkap["fitur"][0, :8].float().cpu().numpy()
        return {"masukan": palm.terapkan_clahe(roi), "fitur": [_ke_0_255(f) for f in fitur], "embedding": e}
