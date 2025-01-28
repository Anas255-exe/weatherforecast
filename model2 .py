import sys
import pickle
import numpy as np

# Load the trained model and label encoder
with open('weather_model.pkl', 'rb') as file:
    model, label_encoder = pickle.load(file)

# Read input features from command-line arguments
features = list(map(float, sys.argv[1:]))
features = np.array(features).reshape(1, -1)

# Predict the weather
prediction = model.predict(features)
result = label_encoder.inverse_transform(prediction)[0]

# Print the result to standard output
print(result)
