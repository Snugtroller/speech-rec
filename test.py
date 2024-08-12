import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# Since you saved the model in a pickle file, we'll load it using pickle
model_dict = pickle.load(open("model.pickle", "rb"))
model = model_dict["model"]

# Load and preprocess test data
test_data_dict = pickle.load(open("data.pickle", "rb"))

test_data = np.asarray(test_data_dict["data"])
test_labels = np.asarray(test_data_dict["labels"])

# Reshape test data to include channel dimension
test_data = test_data.reshape(test_data.shape[0], 128, 128, 1)

# Encode string labels into integers
label_encoder = LabelEncoder()
test_labels = label_encoder.fit_transform(test_labels)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict on a single test sample (optional)
sample_index = 20  # Change this to any index you want to test
sample = np.expand_dims(test_data[sample_index], axis=0)
prediction = model.predict(sample)

# Decode the prediction
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
print(f"Predicted label for sample {sample_index}: {predicted_label[0]}")
print(f"True label for sample {sample_index}: {label_encoder.inverse_transform([test_labels[sample_index]])[0]}")
