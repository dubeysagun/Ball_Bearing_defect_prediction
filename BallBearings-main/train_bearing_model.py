import argparse
import glob
import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

SR = 22050
WINDOW_SEC = 2.0
HOP_SEC = 1.0

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)

    return np.hstack([mfcc_mean, mfcc_std, [centroid, bandwidth, rolloff, zcr, rms], chroma])

def slice_audio(y, sr):
    win = int(WINDOW_SEC * sr)
    hop = int(HOP_SEC * sr)

    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))

    segments = []
    for start in range(0, len(y) - win + 1, hop):
        seg = y[start:start+win]
        segments.append(seg)
    return segments

def load_files(path, label):
    features, labels = [], []
    files = glob.glob(os.path.join(path, "*"))
    for f in files:
        try:
            y, sr = librosa.load(f, sr=SR)
        except:
            y, sr0 = sf.read(f)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = librosa.resample(y, sr0, SR)
            sr = SR

        for seg in slice_audio(y, sr):
            features.append(extract_features(seg, sr))
            labels.append(label)

    return np.array(features), np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--good-dir", default="data/goodSound")
    parser.add_argument("--bad-dir", default="data/badSound")
    parser.add_argument("--model-out", default="model.pkl")
    args = parser.parse_args()

    Xg, yg = load_files(args.good_dir, 0)
    Xb, yb = load_files(args.bad_dir, 1)

    X = np.vstack([Xg, Xb])
    y = np.hstack([yg, yb])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Good", "Faulty"]))

    joblib.dump(model, args.model_out)
    print("Model saved as", args.model_out)

if __name__ == "__main__":
    main()
