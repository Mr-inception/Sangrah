import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModel:
    """
    A comprehensive credit risk modeling class that handles data preprocessing,
    model training, evaluation, and prediction.
    """
    
    def __init__(self, data_path="simulated_credit_risk_data.csv"):
        """
        Initialize the CreditRiskModel.
        
        Args:
            data_path (str): Path to the CSV file containing credit risk data
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        
    def load_data(self):
        """
        Load and validate the credit risk data.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the dataset.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nData types:")
        print(self.df.dtypes)
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        print(f"\nDescriptive statistics:")
        print(self.df.describe())
        
        # Check target distribution
        if 'default' in self.df.columns:
            print(f"\nTarget distribution:")
            print(self.df['default'].value_counts(normalize=True))
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data by separating features and target, scaling features,
        and splitting into train/test sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            # Separate features and target
            if 'default' not in self.df.columns:
                print("Error: 'default' column not found in the dataset.")
                return False
            
            self.X = self.df.drop("default", axis=1)
            self.y = self.df["default"]
            self.feature_names = self.X.columns.tolist()
            
            print(f"Features: {self.feature_names}")
            print(f"Target distribution: {self.y.value_counts().to_dict()}")
            
            # Scale features
            self.X_scaled = self.scaler.fit_transform(self.X)
            
            # Split into train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            
            print(f"Training set size: {self.X_train.shape[0]}")
            print(f"Test set size: {self.X_test.shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return False
    
    def train_models(self):
        """
        Train multiple models for comparison.
        """
        if self.X_train is None:
            print("No preprocessed data. Please run preprocess_data() first.")
            return
        
        print("\n=== MODEL TRAINING ===")
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model
        
        print("Model training completed!")
    
    def evaluate_models(self):
        """
        Evaluate all trained models and print comprehensive metrics.
        """
        if not self.models:
            print("No models trained. Please run train_models() first.")
            return
        
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper().replace('_', ' ')} ---")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            
            results[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC Score: {roc_auc:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion Matrix:\n{cm}")
        
        return results
    
    def get_feature_importance(self, model_name='random_forest'):
        """
        Get feature importance for the specified model.
        
        Args:
            model_name (str): Name of the model to get feature importance for
        
        Returns:
            dict: Feature importance scores
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importance = np.abs(model.coef_[0])
        else:
            print("Model doesn't support feature importance.")
            return None
        
        feature_importance = dict(zip(self.feature_names, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def plot_feature_importance(self, model_name='random_forest'):
        """
        Plot feature importance for the specified model.
        
        Args:
            model_name (str): Name of the model to plot feature importance for
        """
        importance = self.get_feature_importance(model_name)
        
        if importance is None:
            return
        
        plt.figure(figsize=(10, 6))
        features = list(importance.keys())
        scores = list(importance.values())
        
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def predict(self, data, model_name='random_forest'):
        """
        Make predictions on new data.
        
        Args:
            data (pd.DataFrame): New data to predict on
            model_name (str): Name of the model to use for prediction
        
        Returns:
            tuple: (predictions, probabilities)
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None, None
        
        try:
            # Scale the data
            data_scaled = self.scaler.transform(data)
            
            # Make predictions
            model = self.models[model_name]
            predictions = model.predict(data_scaled)
            probabilities = model.predict_proba(data_scaled)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, None
    
    def cross_validate(self, model_name='random_forest', cv=5):
        """
        Perform cross-validation on the specified model.
        
        Args:
            model_name (str): Name of the model to cross-validate
            cv (int): Number of cross-validation folds
        
        Returns:
            dict: Cross-validation results
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='accuracy')
        
        results = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"\n=== CROSS-VALIDATION RESULTS ({model_name}) ===")
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy'] * 2:.4f})")
        print(f"CV Scores: {cv_scores}")
        
        return results


def main():
    """
    Main function to demonstrate the usage of the CreditRiskModel class.
    """
    print("=== CREDIT RISK MODELING ===")
    
    # Initialize the model
    model = CreditRiskModel()
    
    # Load data
    if not model.load_data():
        return
    
    # Explore data
    model.explore_data()
    
    # Preprocess data
    if not model.preprocess_data():
        return
    
    # Train models
    model.train_models()
    
    # Evaluate models
    results = model.evaluate_models()
    
    # Get feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    importance = model.get_feature_importance()
    if importance:
        print("Feature Importance (Random Forest):")
        for feature, score in importance.items():
            print(f"  {feature}: {score:.4f}")
    
    # Cross-validation
    model.cross_validate()
    
    print("\n=== MODELING COMPLETED ===")


if __name__ == "__main__":
    main()

