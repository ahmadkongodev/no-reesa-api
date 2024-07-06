

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import tensorflow as tf
from io import BytesIO
import soundfile as sf
import librosa
import pyttsx3
import os
import speech_recognition as sr

# Initialize FastAPI app
app = FastAPI()

# Dictionaries for translation
dic_moore_english = {
    "nii-yibeugo": "Good morning",
    "nii-zabre": "Good evening",
    "laafi": "I am fine",
    "ni-winiga": "Good afternoon",
    "oub ya laafi": "They are fine",
    "winig-kibare": "how your afternoon is going?",
    "yibeog-kibare": "how your morning is going?",
    "yika laafi": "Did you wake up well?",
    "zabre kibare": "how your evening is going?",
    "zackramba": "and your family?"
}

dic_english_moore = {
    "good morning": "nii-yibeugo",
    "good evening": "nii-zabre",
    "i am fine": "laafi",
    "good afternoon": "ni-winiga",
    "they are fine": "oub ya laafi",
    "how is your afternoon going?": "winig-kibare",
    "how is your morning going?": "yibeog-kibare",
    "did you wake up well?": "yika laafi",
    "how is your evening going?": "zabre kibare",
    "and your family?": "zackramba"
}

# Load the trained model
model = tf.keras.models.load_model("audio_classification_model.h5")

def extract_features(audio_data, sample_rate, n_mfcc=42, max_pad_len=100):
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs.T

classes = ['laafi', 'nii-yibeugo','nii-zabre', 'ni-winiga', 'oub ya laafi','winig-kibare','yibeog-kibare', 'yika laafi', 'zabre kibare', 'zackramba'] 

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/moore_to_english")
async def predict_moore_to_english(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # Load the audio file
        audio_data, sample_rate = sf.read(BytesIO(audio_bytes))

        # Extract features from the audio data
        features = extract_features(audio_data, sample_rate)

        # Reshape to match the input shape of the model
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1) # Add channel dimension

        # Make a prediction
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        moore_prediction = classes[predicted_class]
        english_prediction = dic_moore_english.get(moore_prediction, "Translation not found")

        # Convert text to speech
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'english' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 130)
        audio_path = "english_audio.wav"
        engine.save_to_file(english_prediction, audio_path)
        engine.runAndWait()

        return FileResponse(audio_path, media_type='audio/wav', filename='audio.wav')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.post("/predict/english_to_moore")
async def predict_english_to_moore(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # Recognize speech using SpeechRecognition
        recognizer = sr.Recognizer()
        audio_data = BytesIO(audio_bytes)

        with sr.AudioFile(audio_data) as source:
            audio_content = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_content).lower()
            moore_text = dic_english_moore.get(recognized_text, "Translation not found")
            moore_audio_path = f"moore-audios/{moore_text}.wav"

            if not os.path.exists(moore_audio_path):
                raise HTTPException(status_code=404, detail="Audio file not found for the translation")

        return FileResponse(moore_audio_path, media_type='audio/wav', filename='audio.wav')

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand the audio")
    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Could not request results from the speech recognition service; {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
