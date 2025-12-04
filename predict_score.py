import pickle
import pandas as pd
import numpy as np

# Load the complete model
with open('student_performance_model.pkl', 'rb') as f:
    model_components = pickle.load(f)

model = model_components['model']
scaler = model_components['scaler']
label_encoders = model_components['label_encoders']
feature_names = model_components['feature_names']

def predict_exam_score(student_data):
    """
    Predict exam score for a student
    student_data: dict with student features
    """
    # Convert to DataFrame
    df = pd.DataFrame([student_data])
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    # Ensure correct column order
    df = df.reindex(columns=feature_names)
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return round(prediction, 2)

# Example usage
if __name__ == "__main__":
    sample_student = {
        'Hours_Studied': 25,
        'Attendance': 85,
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 7,
        'Previous_Scores': 80,
        'Motivation_Level': 'High',
        'Internet_Access': 'Yes',
        'Tutoring_Sessions': 2,
        'Family_Income': 'Medium',
        'Teacher_Quality': 'High',
        'School_Type': 'Public',
        'Peer_Influence': 'Positive',
        'Physical_Activity': 4,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'College',
        'Distance_from_Home': 'Near',
        'Gender': 'Female'
    }
    
    predicted_score = predict_exam_score(sample_student)
    print(f"Predicted Exam Score: {predicted_score}")
