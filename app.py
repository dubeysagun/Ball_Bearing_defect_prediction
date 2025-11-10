import streamlit as st
import joblib
import librosa
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment
import tempfile
import os

SR = 22050

# Load trained model
model = joblib.load("model.pkl")

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


def load_audio(uploaded_file):
    """
    Loads .wav, .mp3, .m4a, .flac, .ogg, .aac, and .dat audio safely.
    """
    try:
        st.write(f"📁 Processing file: {uploaded_file.name}")
        st.write(f"📊 File size: {len(uploaded_file.getvalue())} bytes")
        
        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)

        # For .dat files, assume they are raw PCM data
        if uploaded_file.name.endswith('.dat') or uploaded_file.name.endswith('.dat.unknown'):
            try:
                st.write("🔍 Attempting to read as raw PCM data...")
                raw_data = np.frombuffer(file_bytes, dtype=np.int16)
                y = raw_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                st.write("✅ Successfully read as raw PCM data")
                return y, SR
            except Exception as e:
                st.warning(f"⚠️ Raw PCM reading failed: {str(e)}")

        # For M4A files, use pydub first
        if uploaded_file.name.lower().endswith('.m4a'):
            try:
                st.write("🔍 Attempting to read M4A with pydub...")
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name

                # Load with pydub
                audio = AudioSegment.from_file(tmp_path, format="m4a")
                os.unlink(tmp_path)  # Clean up temp file

                # Export as WAV to a new buffer
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)

                # Now load with librosa
                y, sr = librosa.load(wav_buffer, sr=SR, mono=True)
                st.write("✅ Successfully loaded M4A file")
                return y, sr
            except Exception as e:
                st.warning(f"⚠️ Pydub M4A reading failed: {str(e)}")
                file_buffer.seek(0)

        # Try librosa
        try:
            st.write("🔍 Attempting to read with librosa...")
            y, sr = librosa.load(file_buffer, sr=SR, mono=True)
            st.write("✅ Successfully loaded with librosa")
            return y, sr
        except Exception as e:
            st.warning(f"⚠️ Librosa reading failed: {str(e)}")
            file_buffer.seek(0)

        # Try soundfile
        try:
            st.write("🔍 Attempting to read with soundfile...")
            y, sr0 = sf.read(file_buffer)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
            st.write("✅ Successfully loaded with soundfile")
            return y, SR
        except Exception as e:
            st.warning(f"⚠️ Soundfile reading failed: {str(e)}")

        raise ValueError("❌ Could not decode audio. All available methods failed.")
    except Exception as e:
        st.error(f"❌ Error loading audio: {str(e)}")
        raise


st.title("🛠️ Ball Bearing Health Classifier")
st.write("Upload a bearing sound and I will classify it as **Good** or **Faulty**.")

uploaded = st.file_uploader("Upload an audio file", 
                            type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "dat"])

if uploaded:
    st.audio(uploaded)

    try:
        y, sr = load_audio(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    features = extract_features(y, sr).reshape(1, -1)
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success("✅ **Good Bearing** (No fault detected)")
    else:
        st.error("⚠️ **Faulty Bearing** (Potential fault detected)")
