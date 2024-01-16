import os
import numpy as np
import librosa
import tensorflow as tf
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.preprocessing import LabelEncoder

def preprocess(input_data, bg_noise_dir, noise_mixing_ratio=0.1):
    if isinstance(input_data, str):
        wav, _ = librosa.load(input_data, sr=16000, mono=True)
    else:
        wav = input_data

    wav = wav[:48000]
    
    bg_noise_file = np.random.choice(os.listdir(bg_noise_dir))
    bg_noise_path = os.path.join(bg_noise_dir, bg_noise_file)
    bg_noise, _ = librosa.load(bg_noise_path, sr=16000, mono=True)
    
    if bg_noise.shape[0] >= wav.shape[0]:
        start_idx = np.random.randint(0, bg_noise.shape[0] - wav.shape[0])
        bg_noise = bg_noise[start_idx:start_idx+wav.shape[0]]
        wav = (1-noise_mixing_ratio) * wav + noise_mixing_ratio * bg_noise
    else:
        # If the background noise is shorter than the wav, repeat the noise to match the length
        repetitions = int(np.ceil(wav.shape[0] / bg_noise.shape[0]))
        bg_noise = np.tile(bg_noise, repetitions)[:wav.shape[0]]
        wav = (1-noise_mixing_ratio) * wav + noise_mixing_ratio * bg_noise

    mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=20)
    pad_width = 300 - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :300]
    return mfcc


# Update the following directories with your actual dataset directories
dataset_base_path = "C:/Users/abdel/voice_sample"
bg_noise_dir = "C:/Users/abdel/background_noise"


user_directories = {

    'madjid': 'C:/Users/abdel/voice_samples/madjid',
    'lechhub': 'C:/Users/abdel/voice_samples/lechhub',
    'prof_ar': 'C:/Users/abdel/voice_samples/prof ar',
}


audio_file_paths = []
labels = []


for user_id, user_dir in user_directories.items():
    for file in os.listdir(user_dir):
        audio_file_paths.append(os.path.join(user_dir, file))
        labels.append(user_id)


audio_file_paths = np.array(audio_file_paths)
labels = np.array(labels)

X = []

for file_path in audio_file_paths:
    X.append(preprocess(file_path, bg_noise_dir))
X = np.array(X)

# Flatten the MFCC features
X = X.reshape(X.shape[0], -1)

# Expand the dimensions of X to fit the input of the Conv1D layer
X = np.expand_dims(X, axis=-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# Define the 1D CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(user_directories), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseCategoricalAccuracy()])


# Model summary
model.summary()


# Train the model
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)





import pyaudio
import wave

def record_audio(duration=3, filename="recorded.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
def identify_speaker(  bg_noise_dir):
    # Record and preprocess the audio
    record_audio()
    recorded_audio = "recorded.wav"
    preprocessed_audio = preprocess(recorded_audio, bg_noise_dir)
    
    # Reshape the preprocessed audio to fit the input of the model
    input_data = preprocessed_audio.reshape(1, -1, 1)

    # Predict the speaker
    predictions = model.predict(input_data)
    print("Predictions shape:", predictions.shape)

    predicted_label = np.argmax(predictions)
    print("Predicted label:", predicted_label)


    predicted_speaker = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    confidence = predictions[0][predicted_label]  # Corrected line
    return predicted_speaker[0] ,confidence

"""

speaker_name,confidence = identify_speaker( bg_noise_dir)
print("Identified speaker:", speaker_name)
"""