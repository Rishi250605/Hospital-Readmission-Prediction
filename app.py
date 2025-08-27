from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np

# Import XAI module
from explain import create_explainer

# --- Custom Transformers for unpickling ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class AgeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_mapping = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = X.copy()
        if 'age' in X_transformed.columns:
            X_transformed['age_encoded'] = X_transformed['age'].map(self.age_mapping).fillna(-1).astype(int)
            X_transformed = X_transformed.drop(columns=['age'])
        return X_transformed

class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['n_visits'] = (
            X_transformed['n_inpatient'] +
            X_transformed['n_outpatient'] +
            X_transformed['n_emergency']
        )
        X_transformed['proc_med_ratio'] = (
            X_transformed['n_procedures'] / (X_transformed['n_medications'] + 1e-6)
        )
        if 'change' in X_transformed.columns:
            X_transformed = X_transformed.rename(columns={'change': 'change_in_med'})
        return X_transformed

class LabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}
    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            if col in self.label_encoders:
                X_transformed[col] = self.label_encoders[col].transform(X_transformed[col].astype(str))
        return X_transformed

# --- End custom transformers ---

app = Flask(__name__)

# Load the model package
MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'readmission_randomforest.pkl')
print(f"Looking for model at: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)
    pipeline = model_package['pipeline']
    threshold = model_package.get('threshold', 0.4)

# Initialize XAI explainer
print("Initializing XAI explainer...")
try:
    explainer = create_explainer(model_package)
    print("XAI explainer initialized successfully!")
except Exception as e:
    print(f"Warning: XAI explainer failed to initialize: {e}")
    explainer = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    probability = None
    explanation = None
    
    if request.method == 'POST':
        try:
            # Collect form data
            form_data = {
                'age': request.form['age'],
                'time_in_hospital': int(request.form['time_in_hospital']),
                'n_procedures': int(request.form['n_procedures']),
                'n_lab_procedures': int(request.form['n_lab_procedures']),
                'n_medications': int(request.form['n_medications']),
                'n_outpatient': int(request.form['n_outpatient']),
                'n_inpatient': int(request.form['n_inpatient']),
                'n_emergency': int(request.form['n_emergency']),
                'diag_1': request.form['diag_1'],
                'diag_2': request.form['diag_2'],
                'diag_3': request.form['diag_3'],
                'glucose_test': request.form['glucose_test'],
                'A1Ctest': request.form['A1Ctest'],
                'change': request.form['change'],
                'diabetes_med': request.form['diabetes_med']
            }

            # Handle unseen categorical values
            fallback_map = {
                'diag_1': 'Other',
                'diag_2': 'Other',
                'diag_3': 'Other',
                'glucose_test': 'no',
                'A1Ctest': 'no',
                'change': 'no',
                'diabetes_med': 'no'
            }
            
            for col, fallback in fallback_map.items():
                val = form_data.get(col)
                if val is None or val.strip().lower() in ['none', 'null', '']:
                    form_data[col] = fallback

            # Prepare DataFrame for pipeline
            input_df = pd.DataFrame([form_data])

            # Generate prediction with explanation
            if explainer:
                try:
                    explanation = explainer.explain_prediction(input_df)
                    prediction = explanation['prediction']
                    probability = explanation['probability']
                except Exception as e:
                    print(f"XAI explanation failed: {e}")
                    # Fallback to basic prediction
                    prob = pipeline.predict_proba(input_df)[0][1]
                    prediction = "yes" if prob >= threshold else "no"
                    probability = round(prob, 4)
                    explanation = {'error': str(e)}
            else:
                # Basic prediction without explanation
                prob = pipeline.predict_proba(input_df)[0][1]
                prediction = "yes" if prob >= threshold else "no"
                probability = round(prob, 4)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = "Error"
            probability = 0.0
            explanation = {'error': str(e)}

    return render_template('predict.html', 
                         prediction=prediction, 
                         probability=probability,
                         explanation=explanation)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions with explanations
    """
    try:
        data = request.json
        
        # Prepare input DataFrame
        input_df = pd.DataFrame([data])
        
        if explainer:
            explanation = explainer.explain_prediction(input_df)
            return jsonify(explanation)
        else:
            # Basic prediction
            prob = pipeline.predict_proba(input_df)[0][1]
            prediction = "yes" if prob >= threshold else "no"
            
            return jsonify({
                'prediction': prediction,
                'probability': round(float(prob), 4),
                'threshold': threshold,
                'explanation_text': f"Prediction: {prediction} (probability: {prob:.3f}). XAI unavailable."
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    """
    Endpoint to get model information and global feature importance
    """
    if explainer:
        try:
            summary = explainer.get_model_summary()
            return render_template('model_info.html', summary=summary)
        except Exception as e:
            return render_template('model_info.html', error=str(e))
    else:
        return render_template('model_info.html', error="XAI explainer not available")

if __name__ == '__main__':
    app.run(debug=True)