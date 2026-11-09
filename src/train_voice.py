"""Build audio_features.csv and train the voiceprint verification model.

Run from the repo root:  python -m src.train_voice
"""
from src import voice_model
from src.audio_features import build_feature_table

AUDIO_DIR = "data/audio"
FEATURES_PATH = "features/audio_features.csv"
MODEL_PATH = "models/voice_model.pkl"


def main():
    df = build_feature_table(AUDIO_DIR)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Saved {len(df)} rows to {FEATURES_PATH} "
          f"({df['member'].nunique()} members, {df['augmentation'].nunique()} versions per recording)")

    results, confidences = voice_model.evaluate(df)
    summary = results.groupby("model")[["accuracy", "f1", "log_loss"]].mean()
    print("\nLeave-one-phrase-out results (mean of folds):")
    print(summary.round(3))

    best = summary["f1"].idxmax()
    distances = voice_model.genuine_distances(df)
    bundle = voice_model.train_final(df, best, confidences, distances)
    voice_model.save(bundle, MODEL_PATH)
    print(f"\nBest model: {best}. Saved to {MODEL_PATH}")
    print(f"Confidence threshold: {bundle['threshold']:.3f}, "
          f"distance threshold: {bundle['distance_threshold']:.3f}")


if __name__ == "__main__":
    main()
