"""
Student Performance Prediction - Robust ML Pipeline
====================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Basic imports first
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle

def run_ml_pipeline():
    print("STUDENT PERFORMANCE PREDICTION - ML PIPELINE")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and Preprocessing Data...")
    df = pd.read_csv('Student.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df_clean = df.dropna()
    print(f"Dataset shape after removing missing values: {df_clean.shape}")
    
    # Separate features and target
    X = df_clean.drop('Exam_Score', axis=1)
    y = df_clean['Exam_Score']
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {len(categorical_cols)}")
    
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 2: Train multiple algorithms
    print("\nStep 2: Training Multiple ML Algorithms...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    print(f"Training {len(models)} algorithms...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            cv_mean = cv_scores.mean()
            
            results[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV_Mean': cv_mean
            }
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  CV Mean: {cv_mean:.4f}")
            
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Step 3: Results Summary
    print(f"\nStep 3: Results Summary")
    print("=" * 50)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<20} {'R²':<8} {'RMSE':<8}")
    print("-" * 45)
    
    for i, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{i:<4} {name[:19]:<20} {metrics['R2']:.4f}   {metrics['RMSE']:.4f}")
    
    print(f"\nBEST MODEL: {best_name}")
    print(f"BEST R² SCORE: {best_score:.4f}")
    
    # Step 4: Save the best model
    print(f"\nStep 4: Saving Best Model...")
    
    # Save model components
    model_components = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X_encoded.columns),
        'model_name': best_name,
        'r2_score': best_score
    }
    
    with open('student_performance_model.pkl', 'wb') as f:
        pickle.dump(model_components, f)
    
    print("Model saved as: student_performance_model.pkl")
    
    # Step 5: Create deployment script
    print(f"\nStep 5: Creating Deployment Files...")
    
    # Simple deployment script
    deploy_code = '''import pickle
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
'''
    
    with open('predict_score.py', 'w') as f:
        f.write(deploy_code)
    
    print("Deployment script created: predict_score.py")
    
    # Create requirements file
    requirements = '''pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
flask==2.3.3
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("Requirements file created: requirements.txt")
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Files created:")
    print("- student_performance_model.pkl (Complete model)")
    print("- predict_score.py (Prediction script)")
    print("- requirements.txt (Dependencies)")
    print(f"\nBest Model: {best_name}")
    print(f"Best R² Score: {best_score:.4f}")
    print("\nYour model is ready for deployment!")

if __name__ == "__main__":
    run_ml_pipeline()