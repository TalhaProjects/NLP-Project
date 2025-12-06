"""
Student Performance Prediction - Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- LOAD MODEL AND PREPROCESSORS ---
@st.cache_resource
def load_model():
    """Load the trained model and preprocessors from disk."""
    try:
        with open('student_performance_model.pkl', 'rb') as f:
            model_components = pickle.load(f)
        return model_components
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'student_performance_model.pkl' is in the root directory.")
        return None

model_components = load_model()

if model_components:
    model = model_components['model']
    scaler = model_components['scaler']
    label_encoders = model_components['label_encoders']
    feature_names = model_components['feature_names']
    model_name = model_components['model_name']
    model_score = model_components['r2_score']

# --- UI LAYOUT ---
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student's details to predict their exam score.")

# --- MODEL INFORMATION ---
with st.expander("ℹ️ Model Information", expanded=False):
    if model_components:
        st.metric(label="Best Algorithm", value=model_name)
        st.metric(label="Model Accuracy (R² Score)", value=f"{model_score:.4f}")
        st.write("This model predicts student exam scores based on 19 academic and personal factors.")
    else:
        st.warning("Could not load model information.")

# --- INPUT FORM ---
if model_components:
    st.header("Student Details")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.slider("Hours Studied per Week", 0, 50, 20)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        previous_scores = st.slider("Previous Scores (0-100)", 0, 100, 75)
        sleep_hours = st.slider("Sleep Hours per Night", 3, 12, 7)
        tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 10, 1)
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
        
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        
    with col2:
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1)
        parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=0)

        extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], index=0)
        internet_access = st.radio("Internet Access", ["Yes", "No"], index=0)
        learning_disabilities = st.radio("Learning Disabilities", ["No", "Yes"], index=0)
        school_type = st.radio("School Type", ["Public", "Private"], index=0)
        gender = st.radio("Gender", ["Male", "Female"], index=0)

    # --- PREDICTION LOGIC ---
    if st.button("🎯 Predict Exam Score", use_container_width=True):
        # Create a dictionary from the inputs
        student_data = {
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Parental_Involvement': parental_involvement,
            'Access_to_Resources': access_to_resources,
            'Extracurricular_Activities': extracurricular,
            'Sleep_Hours': sleep_hours,
            'Previous_Scores': previous_scores,
            'Motivation_Level': motivation_level,
            'Internet_Access': internet_access,
            'Tutoring_Sessions': tutoring_sessions,
            'Family_Income': family_income,
            'Teacher_Quality': teacher_quality,
            'School_Type': school_type,
            'Peer_Influence': peer_influence,
            'Physical_Activity': physical_activity,
            'Learning_Disabilities': learning_disabilities,
            'Parental_Education_Level': parental_education,
            'Distance_from_Home': distance_from_home,
            'Gender': gender
        }

        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([student_data])

        # Preprocess the data
        try:
            # 1. Encode categorical variables
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    # Handle unseen labels by mapping them to a known category if necessary
                    known_labels = encoder.classes_
                    df[col] = df[col].apply(lambda x: x if x in known_labels else known_labels[0])
                    df[col] = encoder.transform(df[col])
            
            # 2. Ensure correct column order
            df = df.reindex(columns=feature_names, fill_value=0)
            
            # 3. Scale features
            X_scaled = scaler.transform(df)
            
            # 4. Make prediction
            prediction = model.predict(X_scaled)[0]
            
            # Display the result
            st.success(f"**Predicted Exam Score: {prediction:.2f} / 100**")

            # Show a celebratory animation
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please wait for the model to load or check the file path.")
