# Credit Risk Model

A comprehensive credit risk modeling system that uses machine learning to predict credit defaults based on various customer features.

## Features

- **Data Loading and Validation**: Robust data loading with error handling
- **Exploratory Data Analysis**: Comprehensive data exploration and visualization
- **Data Preprocessing**: Feature scaling and train/test splitting
- **Multiple Models**: Logistic Regression and Random Forest classifiers
- **Model Evaluation**: Comprehensive metrics including accuracy, ROC AUC, classification reports, and confusion matrices
- **Feature Importance**: Analysis of feature importance for model interpretability
- **Cross-validation**: Model validation using k-fold cross-validation
- **Prediction Interface**: Easy-to-use prediction interface for new data

## Project Structure

```
├── credit_risk_model.py    # Main credit risk modeling class
├── data.py                 # Data generation script
├── simulated_credit_risk_data.csv  # Sample dataset
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone or download the project files
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the main script to execute the complete credit risk modeling pipeline:

```bash
python credit_risk_model.py
```

This will:
1. Load the credit risk data
2. Perform exploratory data analysis
3. Preprocess the data
4. Train multiple models (Logistic Regression and Random Forest)
5. Evaluate model performance
6. Display feature importance
7. Perform cross-validation

### Advanced Usage

You can also use the `CreditRiskModel` class programmatically:

```python
from credit_risk_model import CreditRiskModel

# Initialize the model
model = CreditRiskModel("your_data.csv")

# Load and explore data
model.load_data()
model.explore_data()

# Preprocess data
model.preprocess_data(test_size=0.2, random_state=42)

# Train models
model.train_models()

# Evaluate models
results = model.evaluate_models()

# Get feature importance
importance = model.get_feature_importance('random_forest')
print(importance)

# Make predictions on new data
new_data = pd.DataFrame({
    'age': [35],
    'monthly_income': [50000],
    'on_time_utility_payments': [0.9],
    'job_stability': [5],
    'social_score': [0.7],
    'ecommerce_monthly_spend': [8000],
    'phone_usage_score': [0.8]
})

predictions, probabilities = model.predict(new_data, 'random_forest')
print(f"Prediction: {predictions[0]}")
print(f"Probability: {probabilities[0]:.4f}")
```

## Data Format

The model expects a CSV file with the following columns:

- `age`: Customer age (integer)
- `monthly_income`: Monthly income (integer)
- `on_time_utility_payments`: Percentage of on-time utility payments (float, 0-1)
- `job_stability`: Years in current job (integer)
- `social_score`: Social activity score (float, 0-1)
- `ecommerce_monthly_spend`: Monthly e-commerce spending (integer)
- `phone_usage_score`: Phone usage score (float, 0-1)
- `default`: Target variable - 1 for default, 0 for no default (integer)

## Model Performance

The system trains and evaluates multiple models:

1. **Logistic Regression**: Linear model for baseline performance
2. **Random Forest**: Ensemble model for better performance and feature importance

Both models are evaluated using:
- Accuracy score
- ROC AUC score
- Classification report (precision, recall, f1-score)
- Confusion matrix

## Key Features

### Error Handling
- Robust error handling for data loading and processing
- Validation of data format and required columns
- Graceful handling of missing data

### Model Interpretability
- Feature importance analysis for Random Forest
- Visualization of feature importance
- Detailed model evaluation metrics

### Flexibility
- Easy to add new models
- Configurable train/test split
- Customizable cross-validation parameters

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- faker: Data generation (for sample data)

## License

This project is open source and available under the MIT License.
