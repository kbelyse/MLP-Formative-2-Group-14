import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.image_features import image_to_features

META_COLS = ["member", "expression", "source_file", "augmentation"]


def embedding_cols(df):
    return [c for c in df.columns if c.startswith("emb_")]


def candidate_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
    }


def evaluate(df, cols=None):
    """Leave-one-expression-out: train on two expressions, test on the third.

    Augmented copies of a photo always stay with that photo, so the test
    images are never seen in training in any form.
    """
    cols = cols or embedding_cols(df)
    X = df[cols].values
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
    X = df[embedding_cols(df)].values
    enc = LabelEncoder()
    y = enc.fit_transform(df["member"])
    pred = np.empty_like(y)
    model = candidate_models()[model_name]
    for held_out in sorted(df["expression"].unique()):
        test = (df["expression"] == held_out).values
        model.fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
    return enc.inverse_transform(y), enc.inverse_transform(pred)


def held_out_similarities(df):
    """Cosine similarity of each held-out photo to the closest training image of
    the same member (genuine) and of every other member (impostor)."""
    E = df[embedding_cols(df)].values
    members = df["member"].values
    genuine, impostor = [], []
    for held_out in sorted(df["expression"].unique()):
        train = (df["expression"] != held_out).values
        test = ((df["expression"] == held_out) & (df["augmentation"] == "original")).values
        sims = E[test] @ E[train].T
        for i, row in enumerate(np.where(test)[0]):
            for m in np.unique(members[train]):
                best = sims[i][members[train] == m].max()
                (genuine if m == members[row] else impostor).append(best)
    return np.array(genuine), np.array(impostor)


def train_final(df, model_name, confidences, genuine, impostor):
    cols = embedding_cols(df)
    X = df[cols].values
    enc = LabelEncoder()
    y = enc.fit_transform(df["member"])
    model = candidate_models()[model_name]
    model.fit(X, y)

    # Check 1: the classifier has to be reasonably sure.
    chance = 1 / len(enc.classes_)
    conf = confidences[model_name]
    threshold = float(np.percentile(conf, 10)) if len(conf) else 0.5
    threshold = max(min(threshold, 0.8), chance + 0.15)

    # Check 2: the face has to look like that member's own photos. The cut-off sits
    # halfway between the most similar pair of different members and the least
    # similar pair of photos of the same member.
    similarity_threshold = float((impostor.max() + genuine.min()) / 2)

    return {
        "model": model,
        "classes": list(enc.classes_),
        "threshold": threshold,
        "reference": X,
        "reference_labels": y,
        "similarity_threshold": similarity_threshold,
    }


def save(bundle, path="models/face_model.pkl"):
    joblib.dump(bundle, path)


def load(path="models/face_model.pkl"):
    return joblib.load(path)


def predict_face(image_path, bundle):
    """Return (member, confidence, similarity). member is None when the face is not recognised."""
    x = image_to_features(image_path)[:, :bundle["reference"].shape[1]]
    proba = bundle["model"].predict_proba(x)[0]
    best = int(proba.argmax())
    confidence = float(proba[best])

    sims = bundle["reference"] @ x[0]
    similarity = float(sims[bundle["reference_labels"] == best].max())

    if confidence < bundle["threshold"] or similarity < bundle["similarity_threshold"]:
        return None, confidence, similarity
    return bundle["classes"][best], confidence, similarity
