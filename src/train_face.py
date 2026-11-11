"""Build image_features.csv and train the face recognition model.

Run from the repo root:  python -m src.train_face
"""
from src import face_model
from src.image_features import build_feature_table

IMAGES_DIR = "data/images"
FEATURES_PATH = "features/image_features.csv"
MODEL_PATH = "models/face_model.pkl"


def main():
    df = build_feature_table(IMAGES_DIR)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Saved {len(df)} rows to {FEATURES_PATH} "
          f"({df['member'].nunique()} members, {df['augmentation'].nunique()} versions per photo)")

    results, confidences = face_model.evaluate(df)
    summary = results.groupby("model")[["accuracy", "f1", "log_loss"]].mean()
    print("\nLeave-one-expression-out results (mean of 3 folds):")
    print(summary.round(3))

    best = summary["f1"].idxmax()
    genuine, impostor = face_model.held_out_similarities(df)
    bundle = face_model.train_final(df, best, confidences, genuine, impostor)
    face_model.save(bundle, MODEL_PATH)
    print(f"\nBest model: {best}. Saved to {MODEL_PATH}")
    print(f"Confidence threshold: {bundle['threshold']:.3f}, "
          f"similarity threshold: {bundle['similarity_threshold']:.3f}")


if __name__ == "__main__":
    main()
