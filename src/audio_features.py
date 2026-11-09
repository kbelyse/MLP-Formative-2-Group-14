from pathlib import Path

import librosa
import numpy as np
import pandas as pd

AUDIO_EXTS = {".wav"}
PHRASES = ["yes_approve", "confirm_transaction"]
SR = 16000
N_MFCC = 20


def load_audio(path, sr=SR):
    """Load as mono at a fixed sample rate and trim leading/trailing silence."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) == 0:
        raise ValueError(f"Audio is silent: {path}")
    return y


def augment(y, sr=SR):
    """Return the original clip plus augmented versions, keyed by name."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.005, len(y)).astype(y.dtype)
    return {
        "original": y,
        "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
        "stretch_fast": librosa.effects.time_stretch(y, rate=1.1),
        "stretch_slow": librosa.effects.time_stretch(y, rate=0.9),
        "noise": y + noise,
        "louder": np.clip(y * 1.5, -1, 1),
        "quieter": y * 0.5,
    }


SCALAR_FEATURES = ["rolloff", "centroid", "bandwidth", "rms", "zcr"]


def extract_features(y, sr=SR):
    """MFCCs + deltas, spectral roll-off/centroid/bandwidth, RMS energy and ZCR (mean and std over time)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    frames = np.vstack([
        mfcc,
        librosa.feature.delta(mfcc),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.rms(y=y),
        librosa.feature.zero_crossing_rate(y),
    ])
    return np.column_stack([frames.mean(axis=1), frames.std(axis=1)]).ravel()


def feature_names():
    rows = ([f"mfcc_{i}" for i in range(N_MFCC)]
            + [f"mfcc_delta_{i}" for i in range(N_MFCC)]
            + SCALAR_FEATURES)
    return [f"{r}_{stat}" for r in rows for stat in ("mean", "std")]


def find_audio(audio_dir):
    """Yield (member, phrase, path) for every wav under audio_dir/<member>/."""
    for member_dir in sorted(Path(audio_dir).iterdir()):
        if not member_dir.is_dir():
            continue
        for path in sorted(member_dir.iterdir()):
            if path.suffix.lower() in AUDIO_EXTS:
                yield member_dir.name, path.stem.lower(), path


def build_feature_table(audio_dir):
    rows = []
    names = feature_names()
    for member, phrase, path in find_audio(audio_dir):
        for aug_name, aug_y in augment(load_audio(path)).items():
            rows.append({
                "member": member,
                "phrase": phrase,
                "source_file": str(path),
                "augmentation": aug_name,
                **dict(zip(names, extract_features(aug_y))),
            })
    return pd.DataFrame(rows)


def audio_to_features(path):
    """Features for a single new recording, used at prediction time (no augmentation)."""
    return extract_features(load_audio(path)).reshape(1, -1)
