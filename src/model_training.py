"""
Model training module for Titanic survival prediction.
Handles model training, evaluation, and comparison.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle
import os

from .config import RF_PARAMS, GB_PARAMS, CV_FOLDS, TEST_SIZE, RANDOM_SEED


class ModelTrainer:
    """Class to handle model training and evaluation."""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(**RF_PARAMS)
        self.gb_model = GradientBoostingClassifier(**GB_PARAMS)
        self.trained_models = {}
        
    def cross_validate_model(self, model, X, y, model_name):
        """Perform cross-validation on a model."""
        scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='accuracy')
        
        print(f"\n{model_name} Cross-Validation Results:")
        print("="*50)
        print(f"CV Scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model."""
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def evaluate_model(self, model, X_val, y_val, model_name):
        """Evaluate model performance."""
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\n{model_name} Validation Performance:")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Died', 'Survived']))
        
        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_val, y_pred)
        }
    
    def get_feature_importance(self, model, feature_cols, top_n=20):
        """Get feature importance from a trained model."""
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models and evaluate if validation set provided."""
        results = {}
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_scores = self.cross_validate_model(self.rf_model, X_train, y_train, "Random Forest")
        self.train_model(self.rf_model, X_train, y_train, "Random Forest")
        results['rf_cv_scores'] = rf_scores
        
        # Train Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb_scores = self.cross_validate_model(self.gb_model, X_train, y_train, "Gradient Boosting")
        self.train_model(self.gb_model, X_train, y_train, "Gradient Boosting")
        results['gb_cv_scores'] = gb_scores
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            results['rf_eval'] = self.evaluate_model(self.rf_model, X_val, y_val, "Random Forest")
            results['gb_eval'] = self.evaluate_model(self.gb_model, X_val, y_val, "Gradient Boosting")
        
        return results
    
    def make_predictions(self, X_test, model_name='Random Forest'):
        """Make predictions using a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        return predictions, probabilities
    
    def make_ensemble_predictions(self, X_test, weights=None):
        """Make ensemble predictions combining multiple models."""
        if weights is None:
            weights = {'Random Forest': 0.5, 'Gradient Boosting': 0.5}
        
        ensemble_proba = np.zeros(len(X_test))
        
        for model_name, weight in weights.items():
            if model_name in self.trained_models:
                _, proba = self.make_predictions(X_test, model_name)
                ensemble_proba += weight * proba
        
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def save_models(self, model_dir):
        """Save trained models to disk."""
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filename = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {filename}")
    
    def load_models(self, model_dir):
        """Load trained models from disk."""
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.trained_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} from {filepath}") 