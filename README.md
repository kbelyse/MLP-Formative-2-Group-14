# MLP Formative 2 - Group 14

## Multimodal Identity Verification and Product Recommendation System

This project combines facial recognition, voice verification and machine-learning-based product recommendations.

A user must first be recognised by their face. Their voice is then checked against the recognised identity. Only when both checks pass does the system retrieve the associated customer record and generate a product recommendation.

## Group members

- Rashid Ali Ntagungira
- Kyla Nyaboke Ochweri
- Sandrine Umugwaneza
- Belyse Kalisa Teta Yamwakate

## Demo video
Unlisted Youtube Video Link; https://youtu.be/syI39fsMboo
[System simulation video](https://drive.google.com/file/d/1PXs7F1UCw0I-TUVWAv9WYofmf42XTlbf/view?usp=sharing)

## Project structure

- `data/raw/` - Customer social profiles and transaction datasets
- `data/processed/` - Merged customer dataset
- `data/images/` - Registered users' facial images
- `data/audio/` - Registered users' voice recordings
- `data/unauthorized/` - Unauthorised authentication samples
- `features/` - Extracted image and audio features
- `notebooks/` - Data preparation, experiments and model evaluation
- `src/` - Face, voice and product model scripts
- `app/` - Integrated command-line application
- `models/` - Trained model files

## Setup

Clone the repository:

```bash
git clone https://github.com/kbelyse/MLP-Formative-2-Group-14.git
cd MLP-Formative-2-Group-14
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The saved models were trained with scikit-learn 1.9.1 and OpenCV 4 (`opencv-python<5`). Both versions are pinned in `requirements.txt`, so installing from it gives a working setup.

## Running the integrated application

The integrated command-line application is located at `app/main.py`.

Run it from the repository root:

```bash
python app/main.py --image data/images/belyse/neutral.jpg --audio data/audio/belyse/yes_approve.wav
```

The application accepts two arguments:

- `--image`: Path to the user's facial image.
- `--audio`: Path to the user's voice recording.

### Authentication and recommendation workflow

1. Facial recognition identifies the user.
2. If the face is not recognised, access is denied.
3. Voice verification checks whether the voice belongs to the recognised user.
4. If the face and voice identities do not match, access is denied.
5. If both checks pass, the system retrieves the simulated customer ID.
6. The product model predicts a product category for that customer.

### Example: authorised user

```bash
python app/main.py --image data/images/belyse/neutral.jpg --audio data/audio/belyse/yes_approve.wav
```

Expected outcome: authentication succeeds and a product recommendation is displayed.

### Example: mismatched identities

```bash
python app/main.py --image data/images/kyla/neutral.jpg --audio data/audio/belyse/yes_approve.wav
```

Expected outcome: access is denied because the face and voice identities do not match.

### Example: unauthorized face

```bash
python app/main.py --image data/unauthorized/images/person_03.jpg --audio data/audio/belyse/yes_approve.wav
```

Expected outcome: access is denied because the face does not belong to a registered member.

### Example: unauthorized voice

```bash
python app/main.py --image data/images/kyla/neutral.jpg --audio data/unauthorized/audio/unauthorized.wav
```

Expected outcome: access is denied because the voice does not match the recognised face.

### Simulated customer ID mapping

The customer datasets contain anonymous customer IDs and do not contain the actual identities of the group members. Therefore, the team assigned four existing customer records to members for demonstration purposes.

| Group member | Simulated customer ID | Expected recommendation |
|---|---:|---|
| Belyse | 187 | Sports |
| Kyla | 114 | Clothing |
| Rashid | 101 | Books |
| Sandrine | 188 | Electronics |

The mapping is a simulation assumption, not a claim that these customer records belong to the actual group members. The application uses `purchase_month=2` for the demonstration.

## Image features and face recognition

Run:

```bash
python -m src.train_face
```

This extracts image features and trains the face recognition model.

The project uses the OpenCV face detection and face recognition ONNX models stored in `models/`.

Detailed experiments and visualisations are available in `notebooks/02_image_processing.ipynb`.

## Audio features and voice recognition

Each registered member has voice recordings in `data/audio/<member>/`.

Run:

```bash
python -m src.train_voice
```

This extracts audio features and trains the voice recognition model.

New recordings can be converted to mono, 16 kHz WAV files using FFmpeg:

```bash
ffmpeg -i recording.m4a -ac 1 -ar 16000 data/audio/<member>/yes_approve.wav
```

Detailed audio processing, spectrograms, augmentations and model evaluation are available in `notebooks/03_audio_processing.ipynb`.

## Customer data and product recommendation

The customer social profile and transaction datasets are cleaned and merged to prepare features for a Random Forest product recommendation model.

Run:

```bash
python -m src.train_product
```

This creates the merged dataset and trained product model.

To generate an individual recommendation:

```python
from src.product_model import predict_product

product, confidence = predict_product(151, purchase_month=2)
print(product, confidence)
```

The model uses ten features:

- `engagement_score`
- `purchase_interest_score`
- `profile_count`
- `previous_purchases`
- `avg_previous_amount`
- `avg_previous_rating`
- `engagement_interest`
- `purchase_month`
- `social_media_platform`
- `review_sentiment`

## Model evaluation and limitations

The product model's reported evaluation results are:

- Accuracy: 0.25
- Macro F1: 0.173
- Log loss: 1.617

The model does not outperform a simple baseline on accuracy or log loss. Consequently, its recommendations should be treated as experimental predictions rather than reliable purchasing advice.

The confidence score returned by the model should not automatically be interpreted as a calibrated probability.

Further details and limitations are documented in Section 18 of `notebooks/01_data_merge_and_product_model.ipynb`.

## Demonstration and testing

The integration was tested using:

- Registered users with matching face and voice samples.
- Mismatched face and voice samples.
- Unregistered facial images.
- Product recommendations for the four simulated customer IDs.

The tests confirmed that matching registered identities can proceed to recommendation, while mismatched and unrecognised identities are rejected.
