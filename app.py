import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

emotion_model = load_model('models/best_model_emotion.h5')
depression_model = load_model('models/best_model_depression.h5')

emotion_labels = ['Angry', 'Calm', 'Fearful', 'Happy', 'Sad']
def extract_features(audio_path):
    X, sample_rate = librosa.load(audio_path,duration=2.5,sr=22050*2,offset=0.5) #, res_type='kaiser_fast'
    features = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30)
    pad_emotion = 216 - features.shape[1]
    pad_depression = 2584 - features.shape[1]

    if pad_emotion > 0:
        emo_features = np.pad(features, [(0, 0), (0, pad_emotion)], mode='constant')
    elif pad_emotion < 0:
        emo_features = features[:,pad_emotion  ]
    else :
        emo_features = features

    if pad_depression > 0:
        dep_features = np.pad(features, [(0, 0), (0, pad_depression)], mode='constant')
    elif pad_depression < 0:
        dep_features = features[:,pad_depression]
    else:
        dep_features = features

    emo_features = np.expand_dims(emo_features, axis = 0)
    dep_features = np.expand_dims(dep_features, axis = 0)

    return emo_features, dep_features

def predict_emotion_and_depression(audio):
    # Extract audio features
    print(audio)
    print(len(audio))
    emo_features, dep_features = extract_features(audio)

    # Predict emotion
    emotion_pred = emotion_model.predict(emo_features)[0]
    print(emotion_pred)
    emotion_index = np.argmax(emotion_pred)
    emotion = emotion_labels[emotion_index]

    # Predict depression
    depression_pred = depression_model.predict(dep_features)[0]
    depression = "Depressed" if depression_pred >= 0.5 else "Not Depressed"

    return emotion, depression

def handler(request):
    if request.method == 'POST':
        # Get the audio data from the request
        audio = request.data  # Replace this with the actual way to access the audio data in the request

        # Make predictions using the models
        emotion, depression = predict_emotion_and_depression(audio)

        # Return the predictions as a response
        response = {
            "emotion": emotion,
            "depression": depression
        }

        return response