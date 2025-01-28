from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Load the trained model and label encoder
with open('D:\coding\java\weatherforcaste\python\weather_model.pkl', 'rb') as file:
    model, label_encoder = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (useful for testing or Java integration)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        features = data.get('features')

        # Ensure features are provided
        if not features:
            return jsonify({'error': 'No features provided'}), 400

        # Convert features to numpy array and predict
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        result = label_encoder.inverse_transform(prediction)[0]

        # Return prediction result
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
6