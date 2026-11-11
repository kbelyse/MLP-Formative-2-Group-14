
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import face_model, voice_model, product_model

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"

MEMBER_TO_CUSTOMER_ID = {
    "belyse": 187,
    "kyla": 114,
    "rashid": 101,
    "sandrine": 188,
}


def load_bundle(filename):
    return joblib.load(MODEL_DIR / filename)


def fix_sklearn_compatibility(bundle):
    pipeline = bundle.get("model")

    if hasattr(pipeline, "steps"):
        for _, estimator in pipeline.steps:
            if isinstance(estimator, LogisticRegression):
                if not hasattr(estimator, "multi_class"):
                    estimator.multi_class = "auto"


def authenticate_and_recommend(image_path, audio_path):
    # Load the saved models and customer data
    face_bundle = load_bundle("face_model.pkl")
    voice_bundle = load_bundle("voice_model.pkl")
    product_bundle = load_bundle("product_model.pkl")

    fix_sklearn_compatibility(face_bundle)
    fix_sklearn_compatibility(voice_bundle)

    profiles = pd.read_csv(
        DATA_DIR / "customer_social_profiles.csv"
    )
    transactions = pd.read_csv(
        DATA_DIR / "customer_transactions.csv"
    )

    # Create the legacy numeric customer ID required by the product model
    if "customer_id_legacy" not in profiles.columns:
        profiles["customer_id_legacy"] = (
            profiles["customer_id_new"]
            .astype(str)
            .str.replace("A", "", regex=False)
            .astype(int)
        )

    # 1. Face recognition
    face_member, face_confidence, face_similarity = (
        face_model.predict_face(str(image_path), face_bundle)
    )

    if face_member is None:
        return {
            "status": "DENIED",
            "reason": "Face not recognised"
        }

    # 2. Voice verification
    voice_authorised = voice_model.verify(
        str(audio_path),
        expected_member=face_member,
        bundle=voice_bundle
    )

    if not voice_authorised:
        return {
            "status": "DENIED",
            "reason": "Voice does not match recognised face",
            "face_member": face_member
        }

    # 3. Customer ID lookup
    customer_id = MEMBER_TO_CUSTOMER_ID.get(face_member)

    if customer_id is None:
        return {
            "status": "AUTHENTICATED_BUT_NO_RECOMMENDATION",
            "member": face_member,
            "reason": "Customer ID mapping is missing"
        }

    # 4. Product recommendation
    product, confidence = product_model.predict_product(
        customer_id,
        model=product_bundle,
        profiles=profiles,
        transactions=transactions,
        purchase_month=2
    )

    return {
        "status": "AUTHORISED",
        "member": face_member,
        "customer_id": customer_id,
        "recommended_product": product,
        "recommendation_confidence": confidence,
        "face_confidence": face_confidence,
        "face_similarity": face_similarity
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Identity Verification and Product Recommendation"
    )
    parser.add_argument("--image", required=True, help="Path to face image")
    parser.add_argument("--audio", required=True, help="Path to voice recording")
    args = parser.parse_args()

    if not Path(args.image).is_file():
        parser.error(f"Image file not found: {args.image}")

    if not Path(args.audio).is_file():
        parser.error(f"Audio file not found: {args.audio}")

    print("\n========== MULTIMODAL IDENTITY SYSTEM ==========")

    result = authenticate_and_recommend(args.image, args.audio)

    print("\nStatus:", result["status"])

    if result["status"] == "AUTHORISED":
        print("Member:", result["member"].title())
        print("Customer ID:", result["customer_id"])
        print("Recommended product:", result["recommended_product"])
        print(
            "Recommendation confidence:",
            f'{result["recommendation_confidence"]:.4f}'
        )
        print("Face confidence:", f'{result["face_confidence"]:.4f}')

    else:
        print("Reason:", result.get("reason", "Access denied"))

    print("\n===============================================")


if __name__ == "__main__":
    main()
