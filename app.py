import os
import librosa
import librosa.display
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import matplotlib.pyplot as plt

# Define augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])

dataset_path = "Dataset"

# Iterate through classes and files
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        print(f"class: {class_folder}")
        for audio_file in os.listdir(class_path):
            audio_path = os.path.join(class_path, audio_file)

            print(f"Processing file: {audio_file}")
            y, sr = librosa.load(audio_path, sr=None)

            # Apply augmentation
            augmented_audio = augment(samples=y, sample_rate=sr)
            augmented_audio_path = os.path.join(class_path, f"aug_{audio_file}")
            sf.write(augmented_audio_path, augmented_audio, sr)
            print(f"Augmented file saved as: {augmented_audio_path}")

            # Extract Mel spectrogram
            S = librosa.feature.melspectrogram(y=augmented_audio, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Plot and display the Mel spectrogram
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title=f'Mel-frequency spectrogram for {audio_file}')
            plt.show()

