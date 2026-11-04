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
