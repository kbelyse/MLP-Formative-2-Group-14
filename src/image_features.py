from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
EXPRESSIONS = ["neutral", "smiling", "surprised"]
FACE_SIZE = 112
EMBEDDING_SIZE = 128

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# YuNet finds the face and its landmarks; SFace turns an aligned face into a 128-number embedding.
_detector = cv2.FaceDetectorYN.create(str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.7)
_recognizer = cv2.FaceRecognizerSF.create(str(MODELS_DIR / "face_recognition_sface_2021dec.onnx"), "")


def load_image(path, max_side=800):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_face(img):
    """Aligned 112x112 crop of the most confident face (eyes levelled using the landmarks).

    Falls back to a centre crop if no face is found.
    """
    h, w = img.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(img)
    if faces is None or len(faces) == 0:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = img[y0:y0 + side, x0:x0 + side]
        return cv2.resize(crop, (FACE_SIZE, FACE_SIZE)), False
    face = faces[faces[:, -1].argmax()]
    return _recognizer.alignCrop(img, face), True


def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment(face):
    """Return the original face crop plus augmented versions, keyed by name."""
    rng = np.random.default_rng(0)
    gray = cv2.cvtColor(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    noise = rng.normal(0, 12, face.shape)
    return {
        "original": face,
        "rotate_15": rotate(face, 15),
        "rotate_-15": rotate(face, -15),
        "flip": cv2.flip(face, 1),
        "grayscale": gray,
        "brighter": cv2.convertScaleAbs(face, alpha=1.0, beta=40),
        "darker": cv2.convertScaleAbs(face, alpha=1.0, beta=-40),
        "blur": cv2.GaussianBlur(face, (5, 5), 0),
        "noise": np.clip(face + noise, 0, 255).astype(np.uint8),
    }


def extract_features(face):
    """Face embedding (SFace) + HSV colour histogram + grayscale intensity histogram."""
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    emb = _recognizer.feature(face).ravel()
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hists = []
    for ch, top in zip(range(3), [180, 256, 256]):
        hist = cv2.calcHist([hsv], [ch], None, [16], [0, top]).ravel()
        hists.append(hist / (hist.sum() + 1e-8))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    gray_hist = gray_hist / (gray_hist.sum() + 1e-8)

    return np.concatenate([emb, *hists, gray_hist])


def feature_names():
    names = [f"emb_{i}" for i in range(EMBEDDING_SIZE)]
    for ch in ["h", "s", "v"]:
        names += [f"hist_{ch}_{i}" for i in range(16)]
    names += [f"hist_gray_{i}" for i in range(32)]
    return names


def find_images(images_dir):
    """Yield (member, expression, path) for every image under images_dir/<member>/."""
    for member_dir in sorted(Path(images_dir).iterdir()):
        if not member_dir.is_dir():
            continue
        for path in sorted(member_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                yield member_dir.name, path.stem.lower(), path


def build_feature_table(images_dir):
    rows = []
    names = feature_names()
    for member, expression, path in find_images(images_dir):
        face, found = detect_face(load_image(path))
        if not found:
            print(f"warning: no face detected in {path}, using centre crop")
        for aug_name, aug_img in augment(face).items():
            feats = extract_features(aug_img)
            rows.append({
                "member": member,
                "expression": expression,
                "source_file": str(path),
                "augmentation": aug_name,
                **dict(zip(names, feats)),
            })
    return pd.DataFrame(rows)


def image_to_features(path):
    """Features for a single new image, used at prediction time (no augmentation)."""
    face, _ = detect_face(load_image(path))
    return extract_features(face).reshape(1, -1)
