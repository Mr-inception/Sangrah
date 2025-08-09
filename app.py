from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from credit_risk_model import CreditRiskModel

app = Flask(__name__)

# Global variables to store the model
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    # Initialize the credit risk model
    credit_model = CreditRiskModel()
    
    # Load and preprocess data
    if credit_model.load_data():
        if credit_model.preprocess_data():
            # Train models
            credit_model.train_models()
            
            # Get the trained models and scaler
            model = credit_model.models['random_forest']
            scaler = credit_model.scaler
            feature_names = credit_model.feature_names
            
            print("Model loaded successfully!")
            return True
    
    print("Failed to load model")
    return False

# Eagerly load the model at startup (compatible with Flask 3.x and WSGI servers)
try:
    load_model()
except Exception as _e:
    # Don't crash import; requests will still return a clear error from /api/model-info
    pass

@app.route('/')
def index():
    """Serve the main webpage"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle credit risk prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame with the input data
        input_data = pd.DataFrame([{
            'age': int(data['age']),
            'monthly_income': int(data['monthly_income']),
            'on_time_utility_payments': float(data['on_time_utility_payments']) / 100,
            'job_stability': int(data['job_stability']),
            'social_score': float(data['social_score']),
            'ecommerce_monthly_spend': int(data['ecommerce_monthly_spend']),
            'phone_usage_score': float(data['phone_usage_score'])
        }])
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Determine risk level and recommendation
        if probability > 0.7:
            risk_level = "High"
            recommendation = "Reject application"
        elif probability > 0.3:
            risk_level = "Medium"
            recommendation = "Review with caution"
        else:
            risk_level = "Low"
            recommendation = "Approve application"
        
        # Format probability as percentage
        probability_pct = f"{probability:.1%}"
        
        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'default_probability': probability_pct,
            'recommendation': recommendation,
            'prediction': int(prediction),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info')
def model_info():
    """Return information about the model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Random Forest',
        'features': ['age', 'monthly_income', 'on_time_utility_payments', 'job_stability', 'social_score', 'ecommerce_monthly_spend', 'phone_usage_score'],
        'status': 'loaded'
    })

if __name__ == '__main__':
    # Local/dev server; in production use a WSGI server (see wsgi.py)
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', '5000'))
    print("Starting Flask application...")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
