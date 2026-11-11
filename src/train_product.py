"""Clean and merge the customer data, then train the product recommendation model.

Run from the repo root:  python -m src.train_product

This repeats the main steps of notebooks/01_data_merge_and_product_model.ipynb
without the plots. The notebook contains the full EDA and model comparison.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROFILES_PATH = "data/raw/customer_social_profiles.csv"
TRANSACTIONS_PATH = "data/raw/customer_transactions.csv"
MERGED_PATH = "data/processed/merged_dataset.csv"
MODEL_PATH = "models/product_model.pkl"

NUMERIC_FEATURES = [
    "engagement_score",
    "purchase_interest_score",
    "profile_count",
    "previous_purchases",
    "avg_previous_amount",
    "avg_previous_rating",
    "engagement_interest",
    "purchase_month",
]
CATEGORICAL_FEATURES = ["social_media_platform", "review_sentiment"]
TARGET = "product_category"


def load_and_clean():
    """Notebook sections 1, 4, 5 and 6: load, clean and match the customer IDs."""
    profiles = pd.read_csv(PROFILES_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    # Section 4: remove exact duplicate profile rows
    profiles_clean = profiles.drop_duplicates().copy()

    # Section 5: convert dates and fill missing ratings with the median
    transactions_clean = transactions.copy()
    transactions_clean["purchase_date"] = pd.to_datetime(transactions_clean["purchase_date"])
    median_rating = transactions_clean["customer_rating"].median()
    transactions_clean["customer_rating"] = transactions_clean["customer_rating"].fillna(median_rating)

    # Section 6: turn A178 into 178 so the IDs match the transactions table
    profiles_clean["customer_id_legacy"] = (
        profiles_clean["customer_id_new"].str.replace("A", "", regex=False).astype(int)
    )

    return profiles_clean, transactions_clean


def merge_tables(profiles_clean, transactions_clean):
    """Notebook sections 7 and 8: one profile row per customer, then an inner merge."""
    profiles_by_customer = profiles_clean.groupby("customer_id_legacy").agg(
        engagement_score=("engagement_score", "mean"),
        purchase_interest_score=("purchase_interest_score", "mean"),
        social_media_platform=("social_media_platform", lambda x: x.mode()[0]),
        review_sentiment=("review_sentiment", lambda x: x.mode()[0]),
        profile_count=("customer_id_new", "size"),
    ).reset_index()

    merged = transactions_clean.merge(
        profiles_by_customer,
        on="customer_id_legacy",
        how="inner",
        validate="many_to_one",
    )
    return merged


def add_features(merged):
    """Notebook section 10: features built only from each customer's earlier purchases."""
    features = merged.sort_values(
        ["customer_id_legacy", "purchase_date", "transaction_id"]
    ).copy()

    customer_groups = features.groupby("customer_id_legacy")

    features["previous_purchases"] = customer_groups.cumcount()
    features["avg_previous_amount"] = customer_groups["purchase_amount"].transform(
        lambda values: values.shift().expanding().mean()
    ).fillna(0)
    features["avg_previous_rating"] = customer_groups["customer_rating"].transform(
        lambda values: values.shift().expanding().mean()
    ).fillna(0)
    features["engagement_interest"] = (
        features["engagement_score"] * features["purchase_interest_score"]
    )
    features["purchase_month"] = features["purchase_date"].dt.month

    return features


def train_and_evaluate(features):
    """Notebook sections 12, 13 and 16: split, train the Simpler Random Forest, and evaluate."""
    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=42
        )),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Macro F1:", round(f1_score(y_test, predictions, average="macro", zero_division=0), 3))
    print("Log loss:", round(log_loss(y_test, probabilities, labels=model.classes_), 3))

    return model


def main():
    profiles_clean, transactions_clean = load_and_clean()
    merged = merge_tables(profiles_clean, transactions_clean)
    features = add_features(merged)

    features.to_csv(MERGED_PATH, index=False)
    print(f"Saved {len(features)} rows to {MERGED_PATH}")

    model = train_and_evaluate(features)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
