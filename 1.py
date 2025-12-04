"""
Student Performance Prediction - Complete ML Pipeline
====================================================
Step 1: Feature Engineering and Preprocessing
Step 2: Train and Test 10+ Different ML Algorithms
Step 3: Identify Best Algorithm and Parameters
Step 4: Save Best Model
Step 5: Deployment Preparation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_score = float('-inf')
        self.results = {}
        
    def load_and_preprocess_data(self, file_path):
        """Step 1: Load and preprocess the dataset"""
        print("Step 1: Loading and Preprocessing Data...")
        
        # Load data
        self.df = pd.read_csv(file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
        # Handle missing values
        self.df = self.df.dropna()
        print(f"Shape after removing missing values: {self.df.shape}")
        
        # Separate features and target
        X = self.df.drop('Exam_Score', axis=1)
        y = self.df['Exam_Score']
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {list(categorical_columns)}")
        
        X_encoded = X.copy()
        
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("Data preprocessing completed!\n")
        
        return X_scaled, y
    
    def evaluate_model(self, model, name):
        """Evaluate model performance"""
        # Train the model
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                  cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_R2_Mean': cv_mean,
            'CV_R2_Std': cv_std
        }
        
        self.results[name] = results
        
        # Check if this is the best model
        if r2 > self.best_score:
            self.best_score = r2
            self.best_model = model
            self.best_model_name = name
            
        return results
    
    def train_all_models(self):
        """Step 2: Train and test 10+ different ML algorithms"""
        print("Step 2: Training Multiple ML Algorithms...")
        
        models = {
            '1. Linear Regression': LinearRegression(),
            
            '2. Ridge Regression': Ridge(alpha=1.0),
            
            '3. Lasso Regression': Lasso(alpha=1.0),
            
            '4. ElasticNet': ElasticNet(alpha=1.0),
            
            '5. Decision Tree': DecisionTreeRegressor(random_state=42),
            
            '6. Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            
            '7. Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            
            '8. Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            
            '9. AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            
            '10. XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            
            '11. LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            
            '12. Support Vector Regressor': SVR(kernel='rbf'),
            
            '13. K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            
            '14. Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(100,), 
                                                   max_iter=500, random_state=42)
        }
        
        print(f"Training {len(models)} different algorithms...\n")
        
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                results = self.evaluate_model(model, name)
                print(f"R² Score: {results['R2']:.4f}")
                print(f"RMSE: {results['RMSE']:.4f}")
                print(f"Cross-validation R² Mean: {results['CV_R2_Mean']:.4f} ± {results['CV_R2_Std']:.4f}")
                print("-" * 50)
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                print("-" * 50)
        
        print("All models trained successfully!\n")
    
    def hyperparameter_tuning(self):
        """Step 3: Hyperparameter tuning for best performing models"""
        print("Step 3: Hyperparameter Tuning for Top Models...")
        
        # Select top 3 models for hyperparameter tuning
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['R2'], reverse=True)[:3]
        
        print("Top 3 models for hyperparameter tuning:")
        for i, (name, results) in enumerate(sorted_results, 1):
            print(f"{i}. {name}: R² = {results['R2']:.4f}")
        print()
        
        # Hyperparameter grids for top models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        best_tuned_model = None
        best_tuned_score = float('-inf')
        best_tuned_name = ""
        
        for name, results in sorted_results:
            model_type = None
            if 'Random Forest' in name:
                model_type = 'Random Forest'
                base_model = RandomForestRegressor(random_state=42)
            elif 'XGBoost' in name:
                model_type = 'XGBoost'
                base_model = xgb.XGBRegressor(random_state=42)
            elif 'Gradient Boosting' in name:
                model_type = 'Gradient Boosting'
                base_model = GradientBoostingRegressor(random_state=42)
            
            if model_type and model_type in param_grids:
                print(f"Tuning hyperparameters for {name}...")
                
                grid_search = GridSearchCV(
                    base_model, 
                    param_grids[model_type],
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # Evaluate tuned model
                y_pred_tuned = grid_search.predict(self.X_test)
                r2_tuned = r2_score(self.y_test, y_pred_tuned)
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Tuned R² Score: {r2_tuned:.4f}")
                print(f"Improvement: {r2_tuned - results['R2']:.4f}")
                
                if r2_tuned > best_tuned_score:
                    best_tuned_score = r2_tuned
                    best_tuned_model = grid_search.best_estimator_
                    best_tuned_name = f"{name} (Tuned)"
                
                print("-" * 50)
        
        # Update best model if tuning improved performance
        if best_tuned_score > self.best_score:
            self.best_model = best_tuned_model
            self.best_score = best_tuned_score
            self.best_model_name = best_tuned_name
            
        print(f"Best model after tuning: {self.best_model_name}")
        print(f"Best R² Score: {self.best_score:.4f}\n")
    
    def save_best_model(self):
        """Step 4: Save the best model"""
        print("Step 4: Saving Best Model...")
        
        # Save the model
        with open('best_student_performance_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save preprocessors
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'r2_score': self.best_score,
            'feature_names': list(self.X_train.columns)
        }
        
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
            
        print(f"Best model saved: {self.best_model_name}")
        print(f"Files saved:")
        print("- best_student_performance_model.pkl")
        print("- scaler.pkl")
        print("- label_encoders.pkl") 
        print("- model_info.pkl\n")
    
    def print_final_results(self):
        """Display final results summary"""
        print("=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        # Sort results by R² score
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['R2'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<25} {'R²':<8} {'RMSE':<8} {'MAE':<8}")
        print("-" * 60)
        
        for i, (name, results) in enumerate(sorted_results, 1):
            model_name = name.split('. ', 1)[1] if '. ' in name else name
            print(f"{i:<4} {model_name[:24]:<25} "
                  f"{results['R2']:.4f}   "
                  f"{results['RMSE']:.4f}   "
                  f"{results['MAE']:.4f}")
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"BEST R² SCORE: {self.best_score:.4f}")
        print("=" * 60)
    
    def create_deployment_script(self):
        """Step 5: Create deployment script for cloud deployment"""
        print("Step 5: Creating Deployment Script...")
        
        deployment_script = '''"""
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
'''

        # Save deployment script
        with open('deploy_app.py', 'w') as f:
            f.write(deployment_script)
            
        # Create requirements.txt
        requirements = '''Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
'''
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
            
        print("Deployment files created:")
        print("- deploy_app.py (Flask API)")
        print("- requirements.txt")
        print("\nDeployment Instructions:")
        print("1. Upload all .pkl files and deploy_app.py to your cloud platform")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Run: python deploy_app.py")
        print("4. API will be available at /predict endpoint")
        
    def run_complete_pipeline(self, file_path='Student.csv'):
        """Run the complete ML pipeline"""
        print("STUDENT PERFORMANCE PREDICTION - COMPLETE ML PIPELINE")
        print("=" * 60)
        
        # Step 1: Data preprocessing
        self.load_and_preprocess_data(file_path)
        
        # Step 2: Train multiple algorithms
        self.train_all_models()
        
        # Step 3: Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Step 4: Save best model
        self.save_best_model()
        
        # Step 5: Create deployment script
        self.create_deployment_script()
        
        # Display final results
        self.print_final_results()
        
        print("\nPipeline completed successfully!")
        print("Your model is ready for deployment!")

# Example usage and demonstration
def example_prediction():
    """Example of how to use the saved model for predictions"""
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    # Load the saved model
    try:
        with open('best_student_performance_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
            
        # Example student data
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
        
        # Preprocess the sample
        df_sample = pd.DataFrame([sample_student])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df_sample.columns:
                df_sample[col] = encoder.transform(df_sample[col])
        
        # Scale features
        X_sample = scaler.transform(df_sample)
        
        # Make prediction
        predicted_score = model.predict(X_sample)[0]
        
        print("Sample Student Profile:")
        for key, value in sample_student.items():
            print(f"  {key}: {value}")
        
        print(f"\nPredicted Exam Score: {predicted_score:.2f}")
        
    except FileNotFoundError:
        print("Model files not found. Please run the complete pipeline first.")

if __name__ == "__main__":
    # Initialize the predictor
    predictor = StudentPerformancePredictor()
    
    # Run the complete pipeline
    predictor.run_complete_pipeline('Student.csv')
    
    # Show example prediction
    example_prediction()
