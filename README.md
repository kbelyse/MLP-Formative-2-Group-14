# MLP Formative 2 - Group 14

User identity and product recommendation system. A user is recognised by their face, the prediction is confirmed with their voice, and then the product recommendation is shown.

## Group members

- Rashid Ali Ntagungira
- Kyla Nyaboke Ochweri
- Sandrine Umugwaneza
- Belyse Kalisa Teta Yamwakate

## Structure

- `data/raw/` - customer_social_profiles.csv and customer_transactions.csv
- `data/processed/` - merged dataset
- `data/images/` - facial images for each member
- `data/audio/` - audio samples for each member
- `data/unauthorized/` - samples used for the unauthorized attempt
- `features/` - image_features.csv and audio_features.csv
- `notebooks/` - Jupyter notebooks
- `src/` - pipeline scripts
- `app/` - command line app
- `models/` - trained models

## Setup

```bash
git clone https://github.com/kbelyse/MLP-Formative-2-Group-14.git
cd MLP-Formative-2-Group-14
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Image features and face model

```bash
python -m src.train_face
```

This creates `features/image_features.csv` and `models/face_model.pkl`. The full steps with plots are in `notebooks/02_image_processing.ipynb`.

## Audio features and voice model

Each member has `data/audio/<member>/yes_approve.wav` and `confirm_transaction.wav`. New recordings (e.g. `.m4a` voice notes) are converted once with ffmpeg (`brew install ffmpeg`):

```bash
ffmpeg -i recording.m4a -ac 1 -ar 16000 data/audio/<member>/yes_approve.wav
```

Then:

```bash
python -m src.train_voice
```

This creates `features/audio_features.csv` and `models/voice_model.pkl`. Waveforms, spectrograms, augmentations and the model evaluation are in `notebooks/03_audio_processing.ipynb`. The unauthorized voice sample goes in `data/unauthorized/audio/`.

## Customer data and product model

The two customer datasets in `data/raw/` are cleaned and merged, and a Random Forest is trained to predict the product category a customer is likely to buy.

```bash
python -m src.train_product
```

This creates `data/processed/merged_dataset.csv` and `models/product_model.pkl`. The cleaning, merge checks, EDA plots, model comparison and limitations are in `notebooks/01_data_merge_and_product_model.ipynb`.

To get a recommendation for a customer:

```python
from src.product_model import predict_product

product, confidence = predict_product(151, purchase_month=2)
```

The model uses 10 inputs, which `predict_product()` builds from the raw data: `engagement_score`, `purchase_interest_score`, `profile_count`, `previous_purchases`, `avg_previous_amount`, `avg_previous_rating`, `engagement_interest`, `purchase_month`, `social_media_platform` and `review_sentiment`.

Test results: accuracy 0.25, macro F1 0.173, log loss 1.617. The model does not beat a simple baseline on accuracy or log loss, so its recommendations are not reliable. See Section 18 of the notebook for the limitations.
If `models/product_model.pkl` fails to load because of a scikit-learn version difference, run `python -m src.train_product` once to recreate it.
