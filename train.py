import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Load data
data_dict = pickle.load(open("data.pickle", "rb"))

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Reshape data to include channel dimension
data = data.reshape(data.shape[0], 128, 128, 1)

# Encode string labels into integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Define model
input_shape = (128, 128, 1)

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Assuming you have 4 classes
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Train model
model.fit(data, labels, epochs=20, batch_size=8, validation_split=0.2)
