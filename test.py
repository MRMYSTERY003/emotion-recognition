#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
import streamlit.components.v1 as components
from sklearn.preprocessing import OneHotEncoder
import warnings
import pandas as pd
import joblib
import speech_recognition as sr
import random
import numpy as np
import scipy.io.wavfile as wav

# Load constants
CAT6 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
CAT3 = ["positive", "negative"]



emotion_map = {
    "Happy": {
        "Optimistic": ["Inspired", "Hopeful"],
        "Trusting": ["Intimate", "Sensitive"],
        "Powerful": ["Courageous", "Creative"],
        "Accepted": ["Respected", "Valued"],
        "Proud": ["Confident", "Successful"],
        "Interested": ["Inquisitive", "Curious"],
        "Content": ["Free", "Joyful"],
        "Playful": ["Cheeky", "Aroused"],
        "Excited": ["Energetic", "Eager"]
    },
    "Surprise": {
            "Startled": ["Shocked", "Dismayed"],
            "Confused": ["Disillusioned", "Perplexed"],
            "Amazed": ["Astonished", "Awe"]
    },
    "Fear": {
            "Scared": ["Helpless", "Frightened"],
            "Anxious": ["Insecure", "Worried"],
            "Weak": ["Worthless", "Insignificant"],
            "Rejected": ["Excluded", "Persecuted"],
            "Threatened": ["Nervous", "Exposed"]
    },
    "Sad": {
            "Vulnerable": ["Fragile", "Victimized"],
            "Despair": ["Grief", "Powerless"],
            "Lonely": ["Abandoned", "Isolated"],
            "Depressed": ["Empty", "Inferior"],
            "Hurt": ["Ashamed", "Inadequate"]
    },
    "Disgust": {
            "Disapproving": ["Judgmental", "Embarrassed"],
            "Disappointed": ["Appalled", "Revolted"],
            "Awful": ["Nauseated", "Detestable"],
            "Avoidant": ["Horrified", "Hesitant"]
    },
    "Angry": {
            "Mad": ["Bitter", "Humiliated"],
            "Aggressive": ["Provoked", "Hostile"],
            "Frustrated": ["Violent", "Furious"],
            "Distant": ["Withdrawn", "Numb"],
            "Critical": ["Skeptical", "Dismissive"]
    }
}






subcategory_probabilities = {
    "Happy": {
        "Optimistic": 0.2,
        "Trusting": 0.1,
        "Powerful": 0.15,
        "Accepted": 0.1,
        "Proud": 0.1,
        "Interested": 0.1,
        "Content": 0.15,
        "Playful": 0.05,
        "Excited": 0.05
    },
    "Surprise": {
            "Startled": 0.4,
            "Confused": 0.3,
            "Amazed": 0.1
    },
    "Fear": {
            "Scared": 0.4,
            "Anxious": 0.3,
            "Weak": 0.05,
            "Rejected": 0.1,
            "Threatened": 0.15
    },
    "Sad": {
            "Vulnerable": 0.3,
            "Despair": 0.2,
            "Lonely": 0.1,
            "Depressed": 0.2,
            "Hurt": 0.2
    },
    "Disgust": {
            "Disapproving": 0.4,
            "Disappointed": 0.3,
            "Awful": 0.1,
            "Avoidant": 0.2
    },
    "Angry": {
            "Mad": 0.3,
            "Aggressive": 0.2,
            "Frustrated": 0.1,
            "Distant": 0.2,
            "Critical": 0.2
    }    
   
}




# Function to predict a subcategory based on a broad emotion
def predict_subcategory(broad_emotion):
    if broad_emotion in subcategory_probabilities:
        subcategory_distribution = subcategory_probabilities[broad_emotion]
        subcategory = random.choices(list(subcategory_distribution.keys()), 
                                     weights=list(subcategory_distribution.values()), 
                                     k=1)[0]
        return subcategory
    return None

# Function to predict a sub-subcategory based on a subcategory
def predict_sub_subcategory(broad_emotion, subcategory):
    if broad_emotion in emotion_map and subcategory in emotion_map[broad_emotion]:
        sub_subcategory = random.choice(emotion_map[broad_emotion][subcategory])
        return sub_subcategory
    return None


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']


# Page settings
st.set_page_config(layout="wide")

max_width = 1000
padding_top = 0
padding_right = "20%"
padding_left = "10%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"
st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
    .audio-wave {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 4rem;
        color: #ff6347;
    }}
    .hidden {{
        display: none;
    }}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def save_audio(file):
    with open(os.path.join("audio", file.name), "wb") as f:
        f.write(file.getbuffer())

@st.cache_data
def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)

@st.cache_data
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio, sr=44100)
    a = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=162)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

def plot_emotions(fig, data6, data3=None, title="Detected emotion", categories6=CAT6, categories3=CAT3):
    color_dict = {
        "fear": "grey",
        "positive": "green",
        "angry": "green",
        "happy": "orange",
        "sad": "purple",
        "negative": "red",
        "disgust": "red",
        "surprise": "lightblue"
    }

    if data3 is None:
        pos = data6[3] + data6[5]
        neg = data6[0] + data6[1] + data6[2] + data6[4]
        data3 = np.array([pos, neg])

    ind = categories6[data6.argmax()]
    color6 = color_dict[ind]

    data6 = list(data6)
    n = len(data6)
    data6 += data6[:1]
    angles6 = [i / float(n) * 2 * np.pi for i in range(n)]
    angles6 += angles6[:1]

    ind = categories3[data3.argmax()]
    color3 = color_dict[ind]

    data3 = list(data3)
    n = len(data3)
    data3 += data3[:1]
    angles3 = [i / float(n) * 2 * np.pi for i in range(n)]
    angles3 += angles3[:1]

    fig.set_facecolor('#d1d1e0')
    ax = plt.subplot(122, polar="True")
    plt.polar(angles6, data6, color=color6)
    plt.fill(angles6, data6, facecolor=color6, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles6[:-1], categories6)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.ylim(0, 1)

    ax = plt.subplot(121, polar="True")
    plt.polar(angles3, data3, color=color3, linewidth=2, linestyle="--", alpha=.8)
    plt.fill(angles3, data3, facecolor=color3, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 6)
    ax.set_theta_direction(-1)
    plt.xticks(angles3[:-1], categories3)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.ylim(0, 1)
    plt.suptitle(title)
    plt.subplots_adjust(top=0.75)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=0.7)





import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def extract_features(data, sr):
    # Initialize result array
    result = np.array([])

    # 1. Zero Crossing Rate (ZCR)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # 2. Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # 3. MFCC (Mel Frequency Cepstral Coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))

    # 4. RMS Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # 5. Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))

    # New Features:

    # 6. Spectral Centroid (Timbre)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, spectral_centroid))

    # 7. Spectral Bandwidth (Timbre)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, spectral_bandwidth))

    # 8. Spectral Roll-off
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, spectral_rolloff))

    # 9. Pitch (Fundamental Frequency, F0)
    f0, _, _ = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_mean = np.mean(f0[~np.isnan(f0)]) if len(f0[~np.isnan(f0)]) > 0 else 0
    result = np.hstack((result, pitch_mean))

    # 10. Pitch Variability (Tone)
    pitch_variability = np.std(f0[~np.isnan(f0)]) if len(f0[~np.isnan(f0)]) > 0 else 0
    result = np.hstack((result, pitch_variability))

    # 11. Jitter (Variability in Pitch)
    def compute_jitter(data, sr):
        pitches, magnitudes = librosa.piptrack(y=data, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches) > 0:
            jitter_value = np.std(np.diff(pitches)) / np.mean(pitches)
            return jitter_value
        return 0

    jitter = compute_jitter(data, sr)
    result = np.hstack((result, jitter))

    # 12. Shimmer (Variability in Amplitude)
    def compute_shimmer(y):
        y_diff = np.diff(y)
        shimmer = np.std(np.abs(y_diff)) / np.mean(np.abs(y_diff))
        return shimmer

    shimmer = compute_shimmer(data)
    result = np.hstack((result, shimmer))

    # 13. Harmonics-to-Noise Ratio (HNR)
    def hnr(y, sr):
        if len(y) == 0:
            raise ValueError("Input audio is empty.")

        # Convert the waveform to a spectrogram (magnitude)
        S = np.abs(librosa.stft(y))

        # Apply HPSS (harmonic-percussive source separation)
        harmonics, noise = librosa.decompose.hpss(S)

        # Calculate HNR value
        mean_harmonics = np.mean(harmonics)
        mean_noise = np.mean(noise)

        if mean_noise == 0:
            hnr_value = 0
        else:
            hnr_value = mean_harmonics / mean_noise

        return hnr_value
    # 14. Formants (F1, F2, F3) - Approximated using spectral peaks
    def get_formants(y, sr):
        fft_spectrum = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), d=1/sr)
        magnitude = np.abs(fft_spectrum)
        peak_indices, _ = find_peaks(magnitude, height=0.1)
        formants = freqs[peak_indices]
        return formants[:3]

    formants = get_formants(data, sr)
    formant_features = np.hstack((formants[0], formants[1], formants[2])) if len(formants) >= 3 else [0, 0, 0]
    result = np.hstack((result, formant_features))

    # 15. Emotional Prosodic Contours (Pitch Contours)
    pitch_contour = np.mean(f0[~np.isnan(f0)]) if len(f0[~np.isnan(f0)]) > 0 else 0
    result = np.hstack((result, pitch_contour))

    # 16. Pauses and Silence Detection
    def compute_silences(y, sr, threshold=0.02):
        intervals = librosa.effects.split(y, top_db=20)
        total_duration = librosa.get_duration(y=y, sr=sr)
        silence_duration = total_duration - sum([(end - start) / sr for start, end in intervals])
        return silence_duration

    silence_duration = compute_silences(data, sr)
    result = np.hstack((result, silence_duration))

    return result





def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6, sr=16000)

    # Extract the features using the updated function
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # Data with noise augmentation
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))

    # Data with stretching and pitching augmentation
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))

    return result


# def extract_features(data, sample_rate):
#     result = np.array([])
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
#     result = np.hstack((result, zcr))

#     stft = np.abs(librosa.stft(data))
#     chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, chroma_stft))

#     mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mfcc))

#     rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
#     result = np.hstack((result, rms))

#     mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
#     result = np.hstack((result, mel))
#     return result





# def get_features(data):
#     data, sample_rate = librosa.load(data, duration=2.5, offset=0.6)

#     res1 = extract_features(data, sample_rate)
#     result = np.array(res1)

#     noise_data = noise(data)
#     res2 = extract_features(noise_data, sample_rate)
#     result = np.vstack((result, res2))

#     new_data = stretch(data)
#     data_stretch_pitch = pitch(new_data, sample_rate)
#     res3 = extract_features(data_stretch_pitch, sample_rate)
#     result = np.vstack((result, res3))

#     return result

def analyze_emotion_combinations(emotions):
    # Define the emotion pairs as sets (unordered)
    emotion_pairs = {
        frozenset(['sad', 'fear']) : 'depression',
        frozenset(['fear', 'surprise']) : 'anxiety',
        frozenset(['fear', 'angry']) : 'PTSD',
        frozenset(['confusion', 'disgust']) : 'dementia',
        frozenset(['sad', 'disgust']) : 'suicidal thoughts',
        frozenset(['angry', 'surprise']) : 'violent outburst'    
    }

    try:
        # Convert input emotions list to a set and use frozenset for matching
        return emotion_pairs[frozenset(emotions)]
    except KeyError:
        return 'None'



# Function to convert speech to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=10)
    try:
        st.write("Recognizing...")
        text = recognizer.recognize_google(audio)
        st.write(f"User: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Could not understand the audio")
        return None
    except sr.RequestError:
        st.write("Could not request results; check your internet connection")
        return None

# Function to generate chatbot response using GPT
# def get_chatbot_response(text, condition):
#     prompt = f"You are a helpful assistant for someone with {condition}. {text}"
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=150,
#         temperature=0.7
#     )
#     response_text = response.choices[0].text.strip()
#     st.write(f"Bot: {response_text}")
#     return response_text

# def get_chatbot_response(text, condition):
#     # Prepare the message for the chat model
#     messages = [
#         {"role": "system", "content": f"You are a helpful assistant for someone with {condition}."},
#         {"role": "user", "content": text}
#     ]
    
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",  # or "gpt-4" depending on availability
#             messages=messages,
#             max_tokens=150,
#             temperature=0.7
#         )
#         response_text = response.choices[0].message['content'].strip()
#         st.write(f"Bot: {response_text}")
#         return response_text
#     except Exception as e:
#         st.write(f"An error occurred: {e}")
#         return None



# # Function to convert text to speech and play it
# def text_to_speech(text):
#     tts = gTTS(text=text, lang='en')
#     filename = "response.mp3"
#     tts.save(filename)
#     playsound.playsound(filename)
#     os.remove(filename)

# Function to run a conversation loop
# def run_conversation(condition):
#     st.markdown("<div class='audio-wave'>🔊</div>", unsafe_allow_html=True)
#     st.write(f"Starting conversation for {condition}. Speak now!")
#     start_time = time.time()
#     duration = 60  # 60 seconds

#     while time.time() - start_time < duration:
#         user_input = recognize_speech()
#         if user_input is None:
#             continue

#         response = get_chatbot_response(user_input, condition)
#         text_to_speech(response)

#     st.write("Conversation ended.")
#     # st.experimental_rerun()
#     st.session_state.reset = True



# def record_audio(duration, filename):
#     # Sample rate in Hertz
#     sample_rate = 44100
#     # Record audio
#     print(f"Recording for {duration} seconds...")
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
#     sd.wait()  # Wait until recording is finished
#     print("Recording complete.")

#     # Save audio to WAV file
#     wav.write(filename, sample_rate, audio_data)
#     print(f"Audio saved to {filename}")

rec = False
state = None


# condition_met = st.button("Trigger Conversation") 
# print(condition_met)



# // Trigger the startConversation() function if the condition is met
# //if ({'true' if condition_met else 'false'}) {{
# //    startConversation();
# //}}

from datetime import datetime

# Get the current date and time

def get_date_time():
    current_datetime = datetime.now()

    # Format the current date and time
    return str(current_datetime.strftime("%Y-%m-%d %H:%M:%S")).replace("-", "").replace(" ","").replace(":","")



def detect_mental_state(emotion_list, mental_state_patterns):
    best_match = None
    max_count = 0

    # Iterate over each mental state and its associated emotion patterns
    for mental_state, pattern in mental_state_patterns.items():
        # Count how many emotions from the current mental state pattern are in the input list
        count = sum(1 for emotion in emotion_list if emotion in pattern)
        
        # If more than 3 emotions match, return the corresponding mental state immediately
        if count > 3:
            return mental_state
        
        # Keep track of the best match (mental state with the highest count)
        if count > max_count:
            max_count = count
            best_match = mental_state
    
    # If no mental state matched more than 3, return the best match with the highest count
    return best_match

# Example usage
mental_state_patterns = {
    "Depression": ["Sad", "Vulnerable", "Hurt", "Despair", "Lonely", "Depressed", "Rejected"],
    "Anxiety": ["Fearful", "Anxious", "Nervous", "Scared", "Helpless", "Worried", "Insecure"],
    "PTSD": ["Fearful", "Shocked", "Avoidant", "Helpless", "Mad", "Bitter", "Dismayed"],
    "Bipolar Disorder": ["Happy", "Sad", "Excited", "Energetic", "Frustrated", "Mad", "Despair"],
    "Dementia": ["Confused", "Perplexed", "Disoriented", "Frustrated", "Sad", "Empty", "Lonely"],
    "Suicidal Thoughts": ["Sad", "Despair", "Hurt", "Empty", "Hopeless", "Powerless", "Ashamed"],
    "Violent Outbursts": ["Angry", "Aggressive", "Frustrated", "Hostile", "Violent", "Furious"]
}




def predict_emotion(x_test, model):
    result = {}
    
    # Predictions on the test set
    pred_test = model.predict(x_test)

    # Normalize predictions for each sample
    y_pred_normalized = pred_test / np.sum(pred_test, axis=1, keepdims=True)

    
    top_5_emotion_indices = np.argsort(y_pred_normalized[0])[::-1][:5]
    top_5_emotion_probs = y_pred_normalized[0][top_5_emotion_indices]
    top_5_emotion_labels = [emotion_labels[idx] for idx in top_5_emotion_indices]
    print(top_5_emotion_labels)
    top_5_temp = top_5_emotion_labels[:3]
    # Format and print the results
    result_string = ", ".join([f"{label} {prob*100:.2f}%" for label, prob in zip(top_5_emotion_labels, top_5_emotion_probs)])
    res = [f"{label} {prob*100:.2f}%" for label, prob in zip(top_5_emotion_labels, top_5_emotion_probs)]
    print(result_string)
    result["top_5" ] = result_string

    print("Prediction for 2 emotions: ")

    broad_emotion = top_5_emotion_labels[0].capitalize()
    predicted_subcategory = predict_subcategory(broad_emotion)
    predicted_sub_subcategory = predict_sub_subcategory(broad_emotion, predicted_subcategory)

    t1 = f"Predicted Subcategory for {broad_emotion}: {predicted_subcategory}\n"
    t2 = f"Predicted Sub-subcategory for {broad_emotion} -> {predicted_subcategory}: {predicted_sub_subcategory}\n"
    top_5_temp.append(predicted_subcategory)
    top_5_temp.append(predicted_sub_subcategory)
    result['mental_combo'] = top_5_temp
    result['sub_1'] = t1 + t2
    result['state'] = predicted_sub_subcategory
    

    print(f"Predicted Subcategory for {broad_emotion}: {predicted_subcategory}")
    print(f"Predicted Sub-subcategory for {broad_emotion} -> {predicted_subcategory}: {predicted_sub_subcategory}\n")

    broad_emotion = top_5_emotion_labels[1].capitalize()
    if res[1].split(" ")[1] != "0.00%":
        predicted_subcategory = predict_subcategory(broad_emotion)
        predicted_sub_subcategory = predict_sub_subcategory(broad_emotion, predicted_subcategory)
        t1 = f"Predicted Subcategory for {broad_emotion}: {predicted_subcategory}\n"
        t2 = f"Predicted Sub-subcategory for {broad_emotion} -> {predicted_subcategory}: {predicted_sub_subcategory}\n"

        print(f"Predicted Subcategory for {broad_emotion}: {predicted_subcategory}")
        print(f"Predicted Sub-subcategory for {broad_emotion} -> {predicted_subcategory}: {predicted_sub_subcategory}")
    else:
        t1 = f"Second emotion contribution is close to 0%"
        t2 = ""
        
        print("Second emotion contribution is close to 0%")
    result['sub_2'] = t1 + t2
        
    print("------------"*10, "\n")
    return result, pred_test



# Main logic to trigger chatbot based on detected condition
def main():
    global rec, state, condition_met
    if 'reset' not in st.session_state:
        st.session_state.reset = False
    st.title("Neu-Free emotion analysis for e-care buddy hearing product")
    st.sidebar.markdown("## Use the menu to navigate on the site")

    menu = ["Upload audio", "Dataset analysis", "About"]
    choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Upload audio":
        st.subheader("Upload audio")
        audio_file = st.file_uploader("Upload audio file", type=['wav'])

        # time.sleep(3)

        audio = audiorecorder("Click to record", "Click to stop recording")

        if len(audio) > 0:
            # To play audio in frontend:
            st.audio(audio.export().read())  

            # To save audio to a file, use pydub export method:
            audio.export(f"./audio/{get_date_time()}.wav", format="wav")

            # To get audio properties, use pydub AudioSegment properties:
            st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

            rec = True
            st.success("Recording completed")

        if audio_file is not None or rec:
            if not rec:
                st.title("Analyzing...")
                file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
                st.write(file_details)
                st.audio(audio_file, format='audio/wav', start_time=0)

                path = os.path.join("audio", audio_file.name)
                save_audio(audio_file)
            else:
                path = f"./audio/{get_date_time()}.wav"

            wav, sr = librosa.load(path, sr=44100)
            Xdb = get_melspec(path)[1]
            fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
            fig.set_facecolor('#d1d1e0')

            plt.subplot(211)
            plt.title("Wave-form")
            librosa.display.waveshow(wav, sr=sr)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            plt.gca().axes.spines["bottom"].set_visible(False)
            plt.gca().axes.set_facecolor('#d1d1e0')

            plt.subplot(212)
            plt.title("Mel-log-spectrogram")
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            st.write(fig)

            model = load_model("model_test.h5")
            print(model.summary())
            #mfccs = get_mfccs(path, model.input_shape[-1])
            # mfccs = mfccs.reshape(1, *mfccs.shape)
            # x_test = mfccs


################################################################
            mfccs = get_mfccs(path, model.input_shape[-1])
            # desired_length = 162
            # padded_x = np.pad(mfccs, ((0, 0), (0, desired_length - mfccs.shape[1]), (0, 0)), mode='constant', constant_values=0)
            # print("==========================in",padded_x.shape)
            new_arr = np.interp(np.linspace(0, 1, 162), np.linspace(0, 1, 128), mfccs.flatten())
            new_arr = new_arr.reshape(162,1)
            new_arr = new_arr.reshape(1, *new_arr.shape)

            print("==========================in", new_arr.shape)
            mfccs = mfccs.reshape(1, *mfccs.shape)
            print("==========================in1::",mfccs.shape)
            # pred = model.predict(new_arr)[0]
            # print("===========================pred:::",pred)

            # ***********************************************
            data_fin, sample_rate_fin = librosa.load(
               str(path))
            data,sample_rate=data_fin,sample_rate_fin
            X = []

            feature = get_features(
                str(path))


########################################################################


            scaler = joblib.load('scaler_new.pkl1')
            x_test = scaler.transform(feature)

            pred_res = predict_emotion(x_test, model)

            result = detect_mental_state(pred_res[0]['mental_combo'], mental_state_patterns)
            print(result)
            title = f"Top 5 emotions: {pred_res[0]['top_5']}"
            st.write(title)
            st.warning(f"Mental State: {result}")
            st.write(pred_res[0]['sub_1'])
            st.write(pred_res[0]['sub_2'])
            # if detected_state != 'None':
            #     st.warning("Caution message: These are experimental results, and not actual results")
            # else:
            st.warning("Caution message: For the experiment purpose only")
            #st.markdown('<div class="custom-warning">Caution message: For the experiment purpose and due to the privacy and security, mental state is not displayed</div>', unsafe_allow_html=True)



            # Features = pd.DataFrame(X)
            # X1 = Features.values

            # x_test = X1
            # scaler = joblib.load('scaler_new.pkl1')

            # # Use the loaded scaler to transform data
            # x_test = scaler.transform(x_test)
            # x_test = np.expand_dims(x_test, axis=2)
            # print('====================>1')
            # #path_checkpoint = "training/cp.ckpt"
            # path_checkpoint = "speech_audio.h5"
            # #model.load_weights(path_checkpoint)
            # y_pred = model.predict(x_test)
            # y_pred_normalized = y_pred / np.sum(y_pred, axis=1, keepdims=True)



            # for i in range(len(y_pred_normalized)):
            #     top_2_emotion_indices = np.argsort(y_pred_normalized[i])[::-1][:2]
            #     top_2_emotion_probs = y_pred_normalized[i][top_2_emotion_indices]
            #     top_2_emotion_labels = [CAT6[idx] for idx in top_2_emotion_indices]

            #     # Format and print the results
            #     result_string = ", ".join(
            #         [f"{label} {prob * 100:.2f}%" for label, prob in zip(top_2_emotion_labels, top_2_emotion_probs)])


            # encoder_file = "encoder_new.npy"
            # encoder_categories = np.load(encoder_file, allow_pickle=True)
            # categories_list = encoder_categories.tolist()[0]
            # print("=================categories_list::",categories_list)
            # new_encoder = OneHotEncoder(categories=[categories_list])
            # new_encoder.fit(np.array(categories_list).reshape(-1, 1))


            # pred = new_encoder.inverse_transform(y_pred)
            # print("======================pred:::",y_pred[0])
    
            # txt = "Detected emotion:" + str(result_string)
            # print("==========================out", txt)
    

            # # Analyze the top 2 emotions
            # detected_state = analyze_emotion_combinations(top_2_emotion_labels)
            # # title = f"{txt}"
            # # st.write(title)


            # Disable UI and show audio wave while chatbot is active

            fig = plt.figure(figsize=(10, 4))
            plot_emotions(data6=pred_res[1][0], fig=fig, title=pred_res[0]['top_5'])
            st.pyplot(fig)


            # run_conversation(detected_state)
            state = pred_res[0]['state']
            # trigger_convo(detected_state)

            components.html(
            f"""
            <div style="overflow-y: auto; height: 300px; border: 1px solid #ccc; padding: 10px;">
                <div id="chatbox" style="height: 250px; overflow-y: auto; color : white"></div>
            </div>

            <script>
            const chatbox = document.getElementById('chatbox');
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            
            let conversationTimeout;
            let endTime;
            
            recognition.onstart = function() {{
                appendMessage('Voice recognition started. Speak now...');
            }};
            
            recognition.onresult = function(event) {{
                const transcript = event.results[0][0].transcript;
                appendMessage('You: ' + transcript);
                console.log(transcript);
            
                getChatbotResponse(transcript).then(response => {{
                    appendMessage('Bot: ' + response);
                    console.log(response);
                    speak(response);
            
                    if (Date.now() >= endTime) {{
                        endConversation();
                    }} else {{
                        recognition.start();
                    }}
                }});
            }};
            
            function startConversation() {{
                appendMessage('Starting a 30-second conversation...');
                endTime = Date.now() + 30000;
                recognition.start();
                conversationTimeout = setTimeout(endConversation, 30000);
            }}
            
            function endConversation() {{
                recognition.stop();
                clearTimeout(conversationTimeout);
                const farewellMessage = "Thank you for the conversation. Goodbye!";
                appendMessage('Bot: ' + farewellMessage);
                speak(farewellMessage);
            }}
            
            function appendMessage(message) {{
                const p = document.createElement('p');
                p.textContent = message;
                chatbox.appendChild(p);
                chatbox.scrollTop = chatbox.scrollHeight;
            }}
            
            async function getChatbotResponse(text) {{
                const response = await fetch('https://api.openai.com/v1/chat/completions', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer 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`
                    }},
                    body: JSON.stringify({{
                        model: "gpt-3.5-turbo",
                        messages: [
                            {{ role: "system", content: "You are a helpful assistant." }},
                            {{ role: "user", content: text }}
                        ],
                        max_tokens: 150,
                        temperature: 0.7
                    }})
                }});

                const data = await response.json();
                return data.choices[0].message.content.trim();
            }}
            function speak(message) {{
                const speech = new SpeechSynthesisUtterance(message);
                speech.pitch = 1.2;
                speech.rate = 1.0;
                speech.volume = 0.8;
                speechSynthesis.speak(speech);
            }}
            
            startConversation();
            </script>
            """,
            height=400
        )


    elif choice == "Dataset analysis":
        st.subheader("Dataset analysis")
        # Implement dataset analysis here

    else:
        st.subheader("About")
        st.info("thiruvikkiramanp@gmail.com")

st.button("Re-run")
if __name__ == '__main__':
    main()



