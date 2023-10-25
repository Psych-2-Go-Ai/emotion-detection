import io
import numpy as np
import pydub
import scipy
from scipy.io import wavfile
from pydub import AudioSegment
import base64
import librosa
import tensorflow as tf

class EndpointHandler():
    
    def __init__(self, path):
        self.emotion_labels = ['Angry', 'Calm', 'Fearful', 'Happy', 'Sad']
        self.emotion_model = tf.keras.models.load_model(f"{path}/models/best_model_emotion.h5")
        self.depression_model = tf.keras.models.load_model(f"{path}/models/best_model_depression.h5")
    
    def __call__(self, input_data):
        audio_base64 = input_data.pop("inputs", input_data)
        audio_features = self.preprocess_audio_data(audio_base64)
        emotion_prediction, depression_prediction = self.perform_emotion_analysis(audio_features)
        return {
            "emotion": emotion_prediction,
            "depression": depression_prediction
        }
    
    def get_mfcc_features(self, features, padding):
        padded_features = padding - features.shape[1]
        if padded_features > 0:
            features = np.pad(features, [(0, 0), (0, padded_features)], mode='constant')
        elif padded_features < 0:
            features = features[:, padded_features:]
        return np.expand_dims(features, axis=0)
    
    def preprocess_audio_data(self, base64_string, duration=2.5, desired_sr=22050*2, offset=0.5):
        # audio_base64 = base64_string.replace("data:audio/webm;codecs=opus;base64,", "")
        audio_bytes = base64.b64decode(base64_string)
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io, format="webm")
        
        byte_io = io.BytesIO()
        audio.export(byte_io, format="wav")
        byte_io.seek(0)

        sample_rate, audio_array = wavfile.read(byte_io)

        audio_array = librosa.resample(audio_array.astype(float), orig_sr=sample_rate, target_sr=desired_sr)
        start_sample = int(offset * desired_sr)
        end_sample = start_sample + int(duration * desired_sr)
        audio_array = audio_array[start_sample:end_sample]

        
        # X, sample_rate = librosa.load(audio_io, duration=duration, sr=desired_sr, offset=offset)
        X = librosa.util.normalize(audio_array)
        return librosa.feature.mfcc(y=X, sr=desired_sr, n_mfcc=30)
    
    def perform_emotion_analysis(self, features, emotion_padding=216, depression_padding=2584):
        emotion_features = self.get_mfcc_features(features, emotion_padding)
        depression_features = self.get_mfcc_features(features, depression_padding)
        emotion_prediction = self.emotion_model.predict(emotion_features)[0]
        emotion_prediction = self.emotion_labels[np.argmax(emotion_prediction)]
        depression_prediction = self.depression_model.predict(depression_features)[0]
        # depression_prediction = "Depressed" if depression_prediction >= 0.5 else "Not Depressed"
        return emotion_prediction, depression_prediction