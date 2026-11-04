import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.image_features import image_to_features

META_COLS = ["member", "expression", "source_file", "augmentation"]


def candidate_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
    }


def evaluate(df):
    """Leave-one-expression-out: train on two expressions, test on the third.

    Augmented copies of a photo always stay with that photo, so the test
    images are never seen in training in any form.
    """
    X = df.drop(columns=META_COLS).values
    enc = LabelEncoder()
    y = enc.fit_transform(df["member"])
    labels = np.arange(len(enc.classes_))

    results, confidences = [], {}
    for name, model in candidate_models().items():
        conf_correct = []
        for held_out in sorted(df["expression"].unique()):
            test = (df["expression"] == held_out).values
            model.fit(X[~test], y[~test])
            proba = model.predict_proba(X[test])
            pred = proba.argmax(axis=1)
            results.append({
                "model": name,
                "held_out": held_out,
                "accuracy": accuracy_score(y[test], pred),
                "f1": f1_score(y[test], pred, average="macro", labels=labels, zero_division=0),
                "log_loss": log_loss(y[test], proba, labels=labels),
            })
            conf_correct += list(proba.max(axis=1)[pred == y[test]])
        confidences[name] = np.array(conf_correct)
    return pd.DataFrame(results), confidences


def held_out_predictions(df, model_name):
    """True and predicted member for every row, each predicted by a model that never saw that photo."""
    X = df.drop(columns=META_COLS).values
    enc = LabelEncoder()
    y = enc.fit_transform(df["member"])
    pred = np.empty_like(y)
    model = candidate_models()[model_name]
    for held_out in sorted(df["expression"].unique()):
        test = (df["expression"] == held_out).values
        model.fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
    return enc.inverse_transform(y), enc.inverse_transform(pred)


def genuine_distances(df):
    """Distance from each held-out photo to the closest training image of the same member."""
    X = df.drop(columns=META_COLS).values
    y = df["member"].values
    dists = []
    for held_out in sorted(df["expression"].unique()):
        train = (df["expression"] != held_out).values
        test = ((df["expression"] == held_out) & (df["augmentation"] == "original")).values
        scaler = StandardScaler().fit(X[train])
        d = cosine_distances(scaler.transform(X[test]), scaler.transform(X[train]))
        for i, row in enumerate(np.where(test)[0]):
            dists.append(d[i][y[train] == y[row]].min())
    return np.array(dists)


def train_final(df, model_name, confidences, distances):
    X = df.drop(columns=META_COLS).values
    enc = LabelEncoder()
    y = enc.fit_transform(df["member"])
    model = candidate_models()[model_name]
    model.fit(X, y)

    # Reject faces the model is unsure about. The threshold sits below most
    # correct predictions from cross-validation but above random guessing.
    chance = 1 / len(enc.classes_)
    conf = confidences[model_name]
    threshold = float(np.percentile(conf, 10)) if len(conf) else 0.5
    threshold = max(threshold, chance + 0.15)

    # Second check: the new face must also be close to that member's own photos.
    scaler = StandardScaler().fit(X)
    return {
        "model": model,
        "classes": list(enc.classes_),
        "threshold": threshold,
        "scaler": scaler,
        "reference": scaler.transform(X),
        "reference_labels": y,
        "distance_threshold": float(np.percentile(distances, 80)),
    }


def save(bundle, path="models/face_model.pkl"):
    joblib.dump(bundle, path)


def load(path="models/face_model.pkl"):
    return joblib.load(path)


def predict_face(image_path, bundle):
    """Return (member, confidence, distance). member is None when the face is not recognised."""
    x = image_to_features(image_path)
    proba = bundle["model"].predict_proba(x)[0]
    best = int(proba.argmax())
    confidence = float(proba[best])

    d = cosine_distances(bundle["scaler"].transform(x), bundle["reference"])[0]
    distance = float(d[bundle["reference_labels"] == best].min())

    if confidence < bundle["threshold"] or distance > bundle["distance_threshold"]:
        return None, confidence, distance
    return bundle["classes"][best], confidence, distance
