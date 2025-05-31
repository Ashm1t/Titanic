"""
Main script for Titanic Survival Prediction.
Orchestrates the entire pipeline from data loading to predictions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.config import RANDOM_SEED, TEST_SIZE
from src.feature_engineering import FeatureEngineer
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.visualization import Visualizer


def main():
    """Main execution function."""
    
    print("="*60)
    print("TITANIC SURVIVAL PREDICTION")
    print("Advanced ML with Social Dynamics Feature Engineering")
    print("="*60)
    
    # Initialize classes
    feature_engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_passenger_ids = test_df['PassengerId']
    
    print(f"   - Training set: {train_df.shape}")
    print(f"   - Test set: {test_df.shape}")
    
    # Feature engineering
    print("\n2. Engineering features...")
    train_df_fe = feature_engineer.engineer_features(train_df)
    test_df_fe = feature_engineer.engineer_features(test_df)
    
    new_features = feature_engineer.get_feature_names()
    print(f"   - Created {len(new_features)} new features")
    
    # Visualize survival patterns
    print("\n3. Creating visualizations...")
    survival_fig = visualizer.plot_survival_analysis(train_df_fe)
    survival_fig.savefig('output/survival_analysis.png', dpi=300, bbox_inches='tight')
    print("   - Saved survival analysis plots")
    
    # Data preprocessing
    print("\n4. Preprocessing data...")
    X_train, y_train, X_test, feature_cols = preprocessor.preprocess_data(train_df_fe, test_df_fe)
    print(f"   - Final feature count: {len(feature_cols)}")
    
    # Create validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Train models
    print("\n5. Training models...")
    results = trainer.train_all_models(X_train_split, y_train_split, X_val_split, y_val_split)
    results['y_val'] = y_val_split  # Store for visualization
    
    # Visualize model comparison
    comparison_fig = visualizer.plot_model_comparison(results)
    comparison_fig.savefig('output/model_comparison.png', dpi=300, bbox_inches='tight')
    print("   - Saved model comparison plots")
    
    # Feature importance
    print("\n6. Analyzing feature importance...")
    rf_importance = trainer.get_feature_importance(trainer.rf_model, feature_cols, top_n=20)
    gb_importance = trainer.get_feature_importance(trainer.gb_model, feature_cols, top_n=20)
    
    visualizer.plot_feature_importance(rf_importance, title="Random Forest")
    import matplotlib.pyplot as plt
    plt.savefig('output/rf_feature_importance.png', dpi=300, bbox_inches='tight')
    
    visualizer.plot_feature_importance(gb_importance, title="Gradient Boosting")
    plt.savefig('output/gb_feature_importance.png', dpi=300, bbox_inches='tight')
    print("   - Saved feature importance plots")
    
    # Retrain on full dataset
    print("\n7. Retraining on full dataset...")
    trainer = ModelTrainer()  # Reset trainer
    trainer.train_all_models(X_train, y_train)
    
    # Make predictions
    print("\n8. Making predictions...")
    rf_pred, rf_proba = trainer.make_predictions(X_test, 'Random Forest')
    gb_pred, gb_proba = trainer.make_predictions(X_test, 'Gradient Boosting')
    ensemble_pred, ensemble_proba = trainer.make_ensemble_predictions(X_test)
    
    predictions_dict = {
        'Random Forest': rf_pred,
        'Gradient Boosting': gb_pred,
        'Ensemble': ensemble_pred
    }
    
    # Create submission summary
    summary_df, summary_fig = visualizer.create_submission_summary(predictions_dict)
    summary_fig.savefig('output/prediction_summary.png', dpi=300, bbox_inches='tight')
    print("\n   Prediction Summary:")
    print(summary_df)
    
    # Create submission files
    print("\n9. Creating submission files...")
    os.makedirs('output', exist_ok=True)
    
    # Random Forest submission
    rf_submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': rf_pred
    })
    rf_submission.to_csv('output/rf_submission.csv', index=False)
    
    # Gradient Boosting submission
    gb_submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': gb_pred
    })
    gb_submission.to_csv('output/gb_submission.csv', index=False)
    
    # Ensemble submission
    ensemble_submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': ensemble_pred
    })
    ensemble_submission.to_csv('output/ensemble_submission.csv', index=False)
    
    print("   - Created submission files in output/")
    
    # Save models
    print("\n10. Saving trained models...")
    trainer.save_models('models')
    
    # Save feature columns for future use
    import pickle
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("   - Saved models and feature columns")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutputs created:")
    print("  - output/survival_analysis.png")
    print("  - output/model_comparison.png")
    print("  - output/rf_feature_importance.png")
    print("  - output/gb_feature_importance.png")
    print("  - output/prediction_summary.png")
    print("  - output/rf_submission.csv")
    print("  - output/gb_submission.csv")
    print("  - output/ensemble_submission.csv")
    print("  - models/random_forest.pkl")
    print("  - models/gradient_boosting.pkl")
    print("  - models/feature_columns.pkl")
    
    # Model agreement analysis
    print(f"\nModel Agreement: {(rf_pred == gb_pred).mean():.2%} of predictions match")
    
    return results, predictions_dict


if __name__ == "__main__":
    results, predictions = main() 