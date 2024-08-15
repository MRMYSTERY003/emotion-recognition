from flask import Flask, render_template, request, redirect, url_for, jsonify
from pydub import AudioSegment
import os
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
import warnings
import pandas as pd
import joblib
import speech_recognition as sr
from gtts import gTTS
import openai
import playsound

# --------------------- support functions ---------------------------------







# Set up OpenAI API Key
openai.api_key = "KEY"
# Load constants
CAT6 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
CAT3 = ["positive", "negative"]



def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    print("got mfcc data 1")
    return (rgbImage, Xdb)

def get_mfccs(audio, limit):
    y, sr = librosa.load(audio, sr=44100)
    a = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=162)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    print("got mfcc data 2")

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
    plt.show()

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

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features(data):
    data, sample_rate = librosa.load(data, duration=2.5, offset=0.6)

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))

    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))

    return result

def analyze_emotion_combinations(emotions):
    emotion_pairs = {
        ("sad", "angry"): "depression",
        ("fear", "surprise"): "anxiety",
        ("confused", "sad"): "dementia",
    }

    emotion_set = set(emotions)
    
    for (emotion1, emotion2), state in emotion_pairs.items():
        if {emotion1, emotion2}.issubset(emotion_set):
            return state

    return "None"

# Function to convert speech to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        # st.write("Listening...")
        print("listerning...")
        audio = recognizer.listen(source)
    try:
        # st.write("Recognizing...")
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        # st.write("Recognizing...")
        print(f"User: {text}")
        return text
    except Exception as e :
        # st.write("Could not understand the audio")
        print("Could not understand the audio")
        return None


# Function to generate chatbot response using GPT
def get_chatbot_response(text, condition):
    prompt = f"You are a helpful assistant for someone with {condition}. {text}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    response_text = response.choices[0].text.strip()
    st.write(f"Bot: {response_text}")
    return response_text

# Function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# Function to run a conversation loop
def run_conversation(condition):
    # st.markdown("<div class='audio-wave'>🔊</div>", unsafe_allow_html=True)
    # st.write(f"Starting conversation for {condition}. Speak now!")
    print("start speaking now..")
    start_time = time.time()
    duration = 60  # 60 seconds

    while time.time() - start_time < duration:
        user_input = recognize_speech()
        if user_input is None:
            continue

        response = get_chatbot_response(user_input, condition)
        text_to_speech(response)

    st.write("Conversation ended.")
    st.experimental_rerun()


# -------------------------------------------------------------------------













app = Flask(__name__)
UPLOAD_FOLDER = 'audios'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part

        if 'audio' not in request.files:
            return jsonify({'message': 'No file part'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        
        # Save the file as a .wav
        filepath = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(filepath)
        print("processing ...")
                
        wav, sr = librosa.load(filepath, sr=44100)
        Xdb = get_melspec(filepath)[1]
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
        # st.write(fig)
        print("audio processing done..")
        fig.savefig("spectrum.png")


        model = load_model("./flask test/speech_audio.h5", compile=False)
        mfccs = get_mfccs(filepath, model.input_shape[-1])
        # mfccs = mfccs.reshape(1, *mfccs.shape)
        # x_test = mfccs


################################################################
        mfccs = get_mfccs(filepath, model.input_shape[-1])

        new_arr = np.interp(np.linspace(0, 1, 162), np.linspace(0, 1, 128), mfccs.flatten())
        new_arr = new_arr.reshape(162,1)
        new_arr = new_arr.reshape(1, *new_arr.shape)

        print("==========================in", new_arr.shape)
        mfccs = mfccs.reshape(1, *mfccs.shape)
        print("==========================in1::",mfccs.shape)


        # ***********************************************
        data_fin, sample_rate_fin = librosa.load(
            str(filepath))
        data,sample_rate=data_fin,sample_rate_fin
        X = []

        feature = get_features(
            str(filepath))
        for ele in feature:
            X.append(ele)

        Features = pd.DataFrame(X)
        X1 = Features.values
        x_test = X1
########################################################################


        scaler = joblib.load('./flask test/scaler.pkl1')
        x_test = scaler.transform(x_test)
        x_test = np.expand_dims(x_test, axis=2)

        model.load_weights("./flask test/speech_audio.h5")
        y_pred = model.predict(x_test)
        y_pred_normalized = y_pred / np.sum(y_pred, axis=1, keepdims=True)
        # print(f" y pred : {y_pred} y pred norm : {y_pred_normalized}")

        top_2_emotion_indices = np.argsort(y_pred_normalized[0])[::-1][:2]
        top_2_emotion_labels = [CAT6[idx] for idx in top_2_emotion_indices]
        detected_state = analyze_emotion_combinations(top_2_emotion_labels)

        title = f"Detected conditions: {detected_state}"
        print(title)
        # st.write(title)

        run_conversation(detected_state)

        fig = plt.figure(figsize=(10, 4))
        plot_emotions(data6=y_pred[0], fig=fig, title=title)
        
        return render_template('index.html', audio_file=audio_file.filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
