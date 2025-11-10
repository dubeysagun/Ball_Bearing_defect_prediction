# Ball Bearings Health Classifier

This project is a machine learning-based diagnostic tool that classifies the health condition of ball bearings using their sound signatures. The system determines whether a bearing is **Healthy** or **Faulty**, based on extracted audio features.

## Features

* Detects bearing condition using recorded audio
* Streamlit-based web interface
* Supports multiple audio formats (WAV, MP3, M4A, FLAC, OGG, AAC, DAT)
* Robust feature extraction (MFCC, Spectral, Chroma)
* Logistic Regression model with balanced training

## System Architecture

Audio Input → Feature Extraction → ML Model → Prediction

Prediction Output:

* ✅ Good Bearing
* ⚠️ Faulty Bearing

## Project Structure

```
.
├── app.py                    # Streamlit web application
├── train_bearing_model.py    # Training script
├── model.pkl                 # Saved trained model
├── requirements.txt          # Dependencies
└── data/
    ├── good/                 # Healthy bearing audio samples
    └── bad/                  # Faulty bearing audio samples
```

## Audio & Model Details

| Component       | Description                                             |
| --------------- | ------------------------------------------------------- |
| Sample Rate     | 22050 Hz                                                |
| Window Size     | 2.0 seconds                                             |
| Hop Length      | 1.0 second                                              |
| Feature Vector  | 47 features (MFCC mean+std, spectral, chroma, ZCR, RMS) |
| Model           | Logistic Regression                                     |
| Split           | 80% Train / 20% Test                                    |
| Class Balancing | Enabled                                                 |

## Installation

```
git clone <repository-url>
cd ball-bearings-classifier
pip install -r requirements.txt
```

## Training the Model

Ensure dataset structure:

```
data/
├── good/
└── bad/
```

Then run:

```
python train_bearing_model.py
```

This generates:

```
model.pkl
```

## Running the Web App

```
streamlit run app.py
```

Upload an audio file and get instant classification.

## Supported Formats

WAV, MP3, M4A, FLAC, OGG, AAC, DAT (raw PCM)

## Dependencies

```
streamlit
librosa
scikit-learn
soundfile
audioread
numpy
joblib
pydub
```

## Future Improvements

* Real-time microphone-based monitoring
* Multiple fault-type classification
* Deep Learning-based spectrogram classifier

## Maintainer

Your Name / Organization
