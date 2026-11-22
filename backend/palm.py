"""palm.py - find the hand and cut the 128x128 palm ROI that the CNN model reads.

The same functions cut the ROI for the training data (training/data.py) and for the webcam,
so the model always sees the same kind of crop. ambil_roi_detail also returns the
intermediate images so the web app can show each step.
"""
import cv2
import numpy as np
import mediapipe as mp

UKURAN = 128  # ROI size in pixels

# Static mode: every request is a single still frame, not a video stream.
_hands = mp.solutions.hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)


def deteksi_tangan(frame_bgr):
    """Return (21 landmarks, MediaPipe handedness label), or None if no hand."""
    hasil = _hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not hasil.multi_hand_landmarks:
        return None
    label = hasil.multi_handedness[0].classification[0].label
    return hasil.multi_hand_landmarks[0].landmark, label


def batas_vertikal(skala):
    """Top and bottom edge of the ROI box, in palm widths below the line through both finger bases.

    skala=1.0 is the journal box: a square from 0.15 to 1.15. Smaller values shrink it evenly.
    Larger values keep the bottom edge at 1.15, which is about where the wrist starts, and
    instead widen the box and raise its top edge toward the finger-base line (never above it).
    """
    if skala <= 1.0:
        tambah = 0.5 * (skala - 1)
        return 0.15 - tambah, 1.15 + tambah
    return max(0.0, 0.15 - 0.5 * (skala - 1)), 1.15


def ambil_roi_detail(frame_bgr, lm, skala=1.0):
    """Cut the palm ROI and return the intermediate steps, or None if it does not fit.

    skala is the box width in palm widths (1.0 = the journal box, a palm-width square).
    See batas_vertikal() for how the height changes with it.
    """
    h, w = frame_bgr.shape[:2]
    p5 = np.array([lm[5].x * w, lm[5].y * h])    # index finger base
    p17 = np.array([lm[17].x * w, lm[17].y * h])  # little finger base
    p0 = np.array([lm[0].x * w, lm[0].y * h])    # wrist

    v = p17 - p5
    jarak = float(np.linalg.norm(v))
    if jarak < 40:  # hand too far or too small
        return None
    sudut = np.degrees(np.arctan2(v[1], v[0]))
    pusat = (p5 + p17) / 2

    # Rotate so the p5-p17 line is horizontal (palm always upright)
    M = cv2.getRotationMatrix2D((float(pusat[0]), float(pusat[1])), float(sudut), 1.0)
    rot = cv2.warpAffine(frame_bgr, M, (w, h))
    p0r = M @ np.array([p0[0], p0[1], 1.0])
    arah = 1 if p0r[1] > pusat[1] else -1  # is the wrist below or above the line?

    s = jarak  # palm width; the journal box is s x s
    atas, bawah = batas_vertikal(skala)  # (0.15, 1.15) at skala=1.0, so the journal values stay exact
    y_mulai = pusat[1] + arah * atas * s
    y_akhir = pusat[1] + arah * bawah * s
    y1, y2 = int(min(y_mulai, y_akhir)), int(max(y_mulai, y_akhir))
    x1, x2 = int(pusat[0] - skala * s / 2), int(pusat[0] + skala * s / 2)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None  # box goes outside the image
    potongan = rot[y1:y2, x1:x2]
    if arah == -1:  # normalize orientation
        potongan = cv2.rotate(potongan, cv2.ROTATE_180)
    roi = cv2.cvtColor(potongan, cv2.COLOR_BGR2GRAY)
    return {
        "rot": rot,
        "kotak": (x1, y1, x2, y2),
        "sudut": float(sudut),
        "potongan": potongan.copy(),  # the colour crop, before grey and the resize to 128x128
        "roi": cv2.resize(roi, (UKURAN, UKURAN)),
        # kept so the results page can draw the same box on the photo before it was rotated
        "putar": M,
        "p5": p5,
        "p17": p17,
    }


def ambil_roi(frame_bgr, lm, skala=1.0):
    """Return only the 128x128 grayscale palm ROI, or None."""
    detail = ambil_roi_detail(frame_bgr, lm, skala)
    return None if detail is None else detail["roi"]


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def terapkan_clahe(roi_gray):
    """Contrast enhancement used before the model (step 5), in training and in the app."""
    return _CLAHE.apply(roi_gray)
