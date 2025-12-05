"""
Student Performance Prediction API
Flask Web Application for Cloud Deployment
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model components
try:
    with open('student_performance_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    
    model = model_components['model']
    scaler = model_components['scaler']
    label_encoders = model_components['label_encoders']
    feature_names = model_components['feature_names']
    model_name = model_components['model_name']
    model_score = model_components['r2_score']
    
    print(f"Model loaded successfully: {model_name}")
    print(f"Model R² Score: {model_score:.4f}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; padding: 15px; background-color: #e8f5e8; border-radius: 4px; }
        .model-info { background-color: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Student Performance Predictor</h1>
        
        <div class="model-info">
            <h3>Model Information</h3>
            <p><strong>Algorithm:</strong> {{ model_name }}</p>
            <p><strong>Accuracy (R² Score):</strong> {{ "%.4f"|format(model_score) }}</p>
            <p><strong>Dataset:</strong> 6,378 students</p>
        </div>
        
        <form id="predictionForm">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <div class="form-group">
                        <label for="hours_studied">Hours Studied per Week:</label>
                        <input type="number" id="hours_studied" name="Hours_Studied" min="0" max="50" value="20" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="attendance">Attendance (%):</label>
                        <input type="number" id="attendance" name="Attendance" min="0" max="100" value="85" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="parental_involvement">Parental Involvement:</label>
                        <select id="parental_involvement" name="Parental_Involvement" required>
                            <option value="Low">Low</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="access_to_resources">Access to Resources:</label>
                        <select id="access_to_resources" name="Access_to_Resources" required>
                            <option value="Low">Low</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="extracurricular">Extracurricular Activities:</label>
                        <select id="extracurricular" name="Extracurricular_Activities" required>
                            <option value="No">No</option>
                            <option value="Yes" selected>Yes</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="sleep_hours">Sleep Hours per Night:</label>
                        <input type="number" id="sleep_hours" name="Sleep_Hours" min="3" max="12" value="7" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="previous_scores">Previous Scores (0-100):</label>
                        <input type="number" id="previous_scores" name="Previous_Scores" min="0" max="100" value="75" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="motivation">Motivation Level:</label>
                        <select id="motivation" name="Motivation_Level" required>
                            <option value="Low">Low</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="internet_access">Internet Access:</label>
                        <select id="internet_access" name="Internet_Access" required>
                            <option value="No">No</option>
                            <option value="Yes" selected>Yes</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="tutoring_sessions">Tutoring Sessions per Month:</label>
                        <input type="number" id="tutoring_sessions" name="Tutoring_Sessions" min="0" max="10" value="1" required>
                    </div>
                </div>
                
                <div>
                    <div class="form-group">
                        <label for="family_income">Family Income:</label>
                        <select id="family_income" name="Family_Income" required>
                            <option value="Low">Low</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="teacher_quality">Teacher Quality:</label>
                        <select id="teacher_quality" name="Teacher_Quality" required>
                            <option value="Low">Low</option>
                            <option value="Medium" selected>Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="school_type">School Type:</label>
                        <select id="school_type" name="School_Type" required>
                            <option value="Public" selected>Public</option>
                            <option value="Private">Private</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="peer_influence">Peer Influence:</label>
                        <select id="peer_influence" name="Peer_Influence" required>
                            <option value="Negative">Negative</option>
                            <option value="Neutral" selected>Neutral</option>
                            <option value="Positive">Positive</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="physical_activity">Physical Activity (hours/week):</label>
                        <input type="number" id="physical_activity" name="Physical_Activity" min="0" max="20" value="3" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="learning_disabilities">Learning Disabilities:</label>
                        <select id="learning_disabilities" name="Learning_Disabilities" required>
                            <option value="No" selected>No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="parental_education">Parental Education Level:</label>
                        <select id="parental_education" name="Parental_Education_Level" required>
                            <option value="High School">High School</option>
                            <option value="College" selected>College</option>
                            <option value="Postgraduate">Postgraduate</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="distance_from_home">Distance from Home:</label>
                        <select id="distance_from_home" name="Distance_from_Home" required>
                            <option value="Near" selected>Near</option>
                            <option value="Moderate">Moderate</option>
                            <option value="Far">Far</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="gender">Gender:</label>
                        <select id="gender" name="Gender" required>
                            <option value="Male" selected>Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <button type="submit">🎯 Predict Exam Score</button>
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
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('result').innerHTML = `
                        <div class="result">
                            <h3>🎉 Prediction Result</h3>
                            <p><strong>Predicted Exam Score: ${result.predicted_exam_score}/100</strong></p>
                            <p>Model: ${result.model_used}</p>
                            <p>Model Accuracy: ${result.model_accuracy}</p>
                        </div>
                    `;
                } else {
                    document.getElementById('result').innerHTML = `
                        <div class="result" style="background-color: #ffebee;">
                            <h3>❌ Error</h3>
                            <p>${result.error}</p>
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `
                    <div class="result" style="background-color: #ffebee;">
                        <h3>❌ Error</h3>
                        <p>Failed to connect to the prediction service.</p>
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with prediction form"""
    if model is None:
        return "Model not loaded. Please check the model files.", 500
    
    return render_template_string(HTML_TEMPLATE, 
                                model_name=model_name, 
                                model_score=model_score)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
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
                except ValueError as e:
                    return jsonify({'error': f'Invalid value for {col}: {data[col]}'}), 400
        
        # Ensure correct column order and handle missing columns
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
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': model_name if model else None,
        'model_score': model_score if model else None
    })

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'api_name': 'Student Performance Predictor',
        'version': '1.0.0',
        'model': model_name if model else None,
        'accuracy': round(model_score, 4) if model else None,
        'features': feature_names if model else None,
        'endpoints': {
            '/': 'Home page with web interface',
            '/predict': 'POST endpoint for predictions',
            '/health': 'Health check',
            '/api/info': 'API information'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)