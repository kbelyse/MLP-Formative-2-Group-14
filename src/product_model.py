"""
Product recommendation model: preprocessing and prediction.

This script prepares a customer's data in the same way as the notebook
(01_data_merge_and_product_model.ipynb) and uses the trained model to
predict the product category the customer is likely to buy.

Usage from another script (for example, the CLI app):

    from product_model import predict_product
    product, confidence = predict_product(178)

Usage from the command line:

    python product_model.py 178
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# File locations. Update these if the files are stored elsewhere.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "product_model.pkl"
PROFILES_PATH = BASE_DIR / "data" / "raw" / "customer_social_profiles.csv"
TRANSACTIONS_PATH = BASE_DIR / "data" / "raw" / "customer_transactions.csv"
# The 10 input columns the model was trained on, in the same order.
FEATURE_COLUMNS = [
    "engagement_score",
    "purchase_interest_score",
    "profile_count",
    "previous_purchases",
    "avg_previous_amount",
    "avg_previous_rating",
    "engagement_interest",
    "purchase_month",
    "social_media_platform",
    "review_sentiment",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(model_path=MODEL_PATH):
    """Load the trained product model (the full pipeline)."""
    return joblib.load(model_path)


def load_data(profiles_path=PROFILES_PATH, transactions_path=TRANSACTIONS_PATH):
    """Load and clean both tables, using the same steps as the notebook."""
    profiles = pd.read_csv(profiles_path)
    transactions = pd.read_csv(transactions_path)

    # Profiles: remove exact duplicates and create a numeric customer ID
    profiles = profiles.drop_duplicates().copy()
    profiles["customer_id_legacy"] = (
        profiles["customer_id_new"].str.replace("A", "", regex=False).astype(int)
    )

    # Transactions: convert dates and fill missing ratings with the median
    transactions = transactions.copy()
    transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"])
    median_rating = transactions["customer_rating"].median()
    transactions["customer_rating"] = transactions["customer_rating"].fillna(median_rating)

    return profiles, transactions


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def normalise_customer_id(customer_id):
    """Accept IDs such as 178, "178" or "A178" and return 178."""
    text = str(customer_id).strip().upper()
    if text.startswith("A"):
        text = text[1:]
    if not text.isdigit():
        raise ValueError(f"Invalid customer ID: {customer_id}")
    return int(text)


def build_customer_features(customer_id, profiles, transactions, purchase_month=None):
    """
    Build the 10 model inputs for one customer.

    Profile features are combined as in the notebook (mean of numeric scores,
    most common platform and sentiment). Purchase-history features use all of
    the customer's recorded purchases, since all of them are in the past when
    a new prediction is made.
    """
    customer_id = normalise_customer_id(customer_id)

    customer_profiles = profiles[profiles["customer_id_legacy"] == customer_id]
    if customer_profiles.empty:
        raise ValueError(f"No profile found for customer {customer_id}")

    past_purchases = transactions[transactions["customer_id_legacy"] == customer_id]
    has_history = not past_purchases.empty

    engagement = customer_profiles["engagement_score"].mean()
    interest = customer_profiles["purchase_interest_score"].mean()

    if purchase_month is None:
        purchase_month = pd.Timestamp.today().month

    features = {
        "engagement_score": engagement,
        "purchase_interest_score": interest,
        "profile_count": len(customer_profiles),
        "previous_purchases": len(past_purchases),
        "avg_previous_amount": past_purchases["purchase_amount"].mean() if has_history else 0,
        "avg_previous_rating": past_purchases["customer_rating"].mean() if has_history else 0,
        "engagement_interest": engagement * interest,
        "purchase_month": int(purchase_month),
        "social_media_platform": customer_profiles["social_media_platform"].mode()[0],
        "review_sentiment": customer_profiles["review_sentiment"].mode()[0],
    }

    return pd.DataFrame([features])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_product(customer_id, model=None, profiles=None, transactions=None,
                    purchase_month=None):
    """
    Predict the product category for one customer.

    Returns (product, confidence), where confidence is the model's
    probability for the predicted category, between 0 and 1.
    """
    if model is None:
        model = load_model()
    if profiles is None or transactions is None:
        profiles, transactions = load_data()

    customer_features = build_customer_features(
        customer_id, profiles, transactions, purchase_month
    )

    product = model.predict(customer_features)[0]
    confidence = model.predict_proba(customer_features).max()

    return product, float(confidence)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python product_model.py <customer_id>")
        sys.exit(1)

    try:
        product, confidence = predict_product(sys.argv[1])
        print(f"Predicted product: {product} (confidence: {confidence:.0%})")
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        sys.exit(1)
