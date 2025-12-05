"""
Student Performance Prediction API - Vercel Compatible
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to access model files
sys.path.append('..')

app = Flask(__name__)

# Load the model components with error handling
model = None
scaler = None
label_encoders = None
feature_names = None
model_name = "Model not loaded"
model_score = 0

try:
    # Try to load from current directory first, then parent
    model_paths = ['student_performance_model.pkl', '../student_performance_model.pkl']
    
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                model_components = pickle.load(f)
            
            model = model_components['model']
            scaler = model_components['scaler']
            label_encoders = model_components['label_encoders']
            feature_names = model_components['feature_names']
            model_name = model_components['model_name']
            model_score = model_components['r2_score']
            
            print(f"Model loaded successfully from {path}: {model_name}")
            break
        except FileNotFoundError:
            continue
    
    if model is None:
        print("Warning: Model file not found in any location")
    
except Exception as e:
    print(f"Error loading model: {e}")

# Simplified HTML template for Vercel
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; text-align: center; }
        .form-group { margin: 10px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
        button:hover { background: #45a049; }
        .result { margin: 15px 0; padding: 15px; background: #e8f5e8; border-radius: 4px; }
        .error { background: #ffebee; color: #c62828; }
        .info { background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Student Performance Predictor</h1>
        
        <div class="info">
            <strong>Model:</strong> {{ model_name }}<br>
            <strong>Accuracy:</strong> {{ "%.4f"|format(model_score) }}
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label>Hours Studied per Week:</label>
                <input type="number" name="Hours_Studied" value="20" min="0" max="50" required>
            </div>
            
            <div class="form-group">
                <label>Attendance (%):</label>
                <input type="number" name="Attendance" value="85" min="0" max="100" required>
            </div>
            
            <div class="form-group">
                <label>Previous Scores:</label>
                <input type="number" name="Previous_Scores" value="75" min="0" max="100" required>
            </div>
            
            <div class="form-group">
                <label>Sleep Hours:</label>
                <input type="number" name="Sleep_Hours" value="7" min="3" max="12" required>
            </div>
            
            <div class="form-group">
                <label>Parental Involvement:</label>
                <select name="Parental_Involvement" required>
                    <option value="Low">Low</option>
                    <option value="Medium" selected>Medium</option>
                    <option value="High">High</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Access to Resources:</label>
                <select name="Access_to_Resources" required>
                    <option value="Low">Low</option>
                    <option value="Medium" selected>Medium</option>
                    <option value="High">High</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Motivation Level:</label>
                <select name="Motivation_Level" required>
                    <option value="Low">Low</option>
                    <option value="Medium" selected>Medium</option>
                    <option value="High">High</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Family Income:</label>
                <select name="Family_Income" required>
                    <option value="Low">Low</option>
                    <option value="Medium" selected>Medium</option>
                    <option value="High">High</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>School Type:</label>
                <select name="School_Type" required>
                    <option value="Public" selected>Public</option>
                    <option value="Private">Private</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Gender:</label>
                <select name="Gender" required>
                    <option value="Male" selected>Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            
            <!-- Add remaining fields with default values -->
            <input type="hidden" name="Extracurricular_Activities" value="Yes">
            <input type="hidden" name="Internet_Access" value="Yes">
            <input type="hidden" name="Tutoring_Sessions" value="1">
            <input type="hidden" name="Teacher_Quality" value="Medium">
            <input type="hidden" name="Peer_Influence" value="Neutral">
            <input type="hidden" name="Physical_Activity" value="3">
            <input type="hidden" name="Learning_Disabilities" value="No">
            <input type="hidden" name="Parental_Education_Level" value="College">
            <input type="hidden" name="Distance_from_Home" value="Near">
            
            <button type="submit">🎯 Predict Score</button>
        </form>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            
            for (let [key, value] of formData.entries()) {
                if (['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity'].includes(key)) {
                    data[key] = parseInt(value);
                } else {
                    data[key] = value;
                }
            }
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('result').innerHTML = 
                        '<div class="result"><h3>🎉 Predicted Score: ' + result.predicted_exam_score + '/100</h3></div>';
                } else {
                    document.getElementById('result').innerHTML = 
                        '<div class="result error"><h3>❌ Error: ' + result.error + '</h3></div>';
                }
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    '<div class="result error"><h3>❌ Connection Error</h3></div>';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template_string(HTML_TEMPLATE, 
                                model_name=model_name, 
                                model_score=model_score)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded - please check model files'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    return jsonify({'error': f'Invalid value for {col}'}), 400
        
        # Ensure correct column order
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            'predicted_exam_score': round(prediction, 2),
            'model_used': model_name,
            'model_accuracy': round(model_score, 4)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Vercel compatibility
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)