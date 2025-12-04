"""
Student Performance Prediction API
Deployment Script for Cloud (e.g., PythonAnywhere, Heroku, etc.)
"""

import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and preprocessors
with open('best_student_performance_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

@app.route('/')
def home():
    return """
    <h1>Student Performance Prediction API</h1>
    <p>POST to /predict with student data to get exam score prediction</p>
    <p>Model: """ + model_info['model_name'] + """</p>
    <p>R² Score: """ + str(round(model_info['r2_score'], 4)) + """</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            'predicted_exam_score': round(prediction, 2),
            'model_used': model_info['model_name'],
            'model_accuracy': round(model_info['r2_score'], 4)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
