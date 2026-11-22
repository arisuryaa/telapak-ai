import cv2
import numpy as np


class PengenalPalsu:
    """Same interface as model.Pengenal, without torch: the embedding is the ROI shrunk to 8x16."""

    def __init__(self, versi="palsu-1", batas=0.4):
        self.versi = versi
        self.batas = batas

    def embedding_banyak(self, roi_list):
        e = np.stack([cv2.resize(r, (16, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for r in roi_list])
        e -= e.mean(axis=1, keepdims=True)
        return (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-6)).astype(np.float32)

    def embedding(self, roi):
        return self.embedding_banyak([roi])[0]

    def detail(self, roi):
        kecil = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        return {"masukan": roi.copy(), "fitur": [kecil.copy() for _ in range(8)], "embedding": self.embedding(roi)}
