import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from fastapi import FastAPI
import os
import numpy as np
from scipy.io.wavfile import read
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile
from fastapi import Form
from fastapi import UploadFile, Form
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import wave
import os

app = FastAPI()


origins = [
    "*",
]

# Add middlewares to the origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# Function to extract spectrogram features from audio file
def extract_features(file_path, n_mels=128, n_fft=2048, hop_length=512):
    signal, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

    # Define a function to load data from the given path
def load_data(path):
    features = []
    labels = []
    for speaker_folder in os.listdir(path):
        speaker_path = os.path.join(path, speaker_folder)
        for audio_file in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, audio_file)
            features.append(extract_features(file_path))
            labels.append(speaker_folder)
    return np.array(features), np.array(labels)

def train_model():

    # Load data
    data_path = "samples"
    X, y = load_data(data_path)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Add channel dimension for CNN
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for i in range(10):
        # Train model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc}")

    # Save the model
    model.save("speaker_identification_model.h5")


def save_file(input_file, person_name):
    # Input .wav file

    # Output folder to save the segments
    output_folder = f"samples/{person_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Duration of each segment in milliseconds (0.01 seconds)
    segment_duration = 2000

    # Total duration of the audio in milliseconds
    total_duration = len(audio)

    # Calculate the number of segments
    num_segments = total_duration // segment_duration

    # Split the audio into segments
    for i in range(num_segments):
        # Calculate the start and end time of the segment
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        # Extract the segment
        segment = audio[start_time:end_time]
        
        # Save the segment to a new .wav file
        output_file = os.path.join(output_folder, f"{i}.wav")
        segment.export(output_file, format="wav")

    print(f"Split {input_file} into {num_segments} segments in {output_folder}.")



def save_file_test(input_file, person_name):
    # Input .wav file

    # Output folder to save the segments
    output_folder = f"test/"
    os.makedirs(output_folder, exist_ok=True)

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Duration of each segment in milliseconds (0.01 seconds)
    segment_duration = 2000

    # Total duration of the audio in milliseconds
    total_duration = len(audio)

    # Calculate the number of segments
    num_segments = total_duration // segment_duration

    # Split the audio into segments
    for i in range(num_segments):
        # Calculate the start and end time of the segment
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        # Extract the segment
        segment = audio[start_time:end_time]

        if i==0:

            # Save the segment to a new .wav file
            output_file = os.path.join(output_folder, f"{i}.wav")
            segment.export(output_file, format="wav")

        break

 

@app.post("/record_audio_train")
async def record_audio_train(
    file: UploadFile,
    person_name: str = Form(...),
):
    print(person_name)
    try:
        file_path = f"training_set/sample.wav"
        with open(file_path, "wb") as f:
            file_content = await file.read()
            print(file_content)
            f.write(file_content)
            f.close()
        
        save_file(file_path, person_name)

        return {"message": "File saved successfully"}
    except Exception as e:
        return {"message": e.args}



@app.post("/record_audio_test")
async def record_audio_train(
    file: UploadFile,
    person_name: str = Form(...),
):
    print(person_name)
    try:
        file_path = f"testing_set/sample.wav"
        with open(file_path, "wb") as f:
            file_content = await file.read()
            print(file_content)
            f.write(file_content)
            f.close()
        save_file_test(file_path, person_name)
        return {"message": "File saved successfully"}
    except Exception as e:
        return {"message": e.args}


@app.post("/train_model")
async def train_model_():
    train_model()
    return {"message": "Model trained successfully"}


def predict_person():
    from tensorflow.keras.models import load_model
    import librosa
    import numpy as np

    # Load the saved model
    model = load_model('speaker_identification_model.h5')

    # Function to extract spectrogram features from audio file
    def extract_features(file_path, n_mels=128, n_fft=2048, hop_length=512):
        signal, sr = librosa.load(file_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    # Example audio file for prediction
    sample_file_path = "test/0.wav"
    sample_features = extract_features(sample_file_path)
    sample_features = np.expand_dims(sample_features, axis=-1)  # Add channel dimension for CNN
    sample_features = np.expand_dims(sample_features, axis=0)   # Add batch dimension

    # Make prediction
    predictions = model.predict(sample_features)
    predicted_speaker_index = np.argmax(predictions)
    print(predictions)
     # Load data
    data_path = "samples"
    X, y = load_data(data_path)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    predicted_speaker = label_encoder.inverse_transform([predicted_speaker_index])[0]
    print("Predicted Speaker:", predicted_speaker)
    return predicted_speaker

@app.post("/test_model")
async def predict_person_():

    predicted_person=predict_person()

    return {"speaker": predicted_person}