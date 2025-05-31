# Titanic Survival Prediction
## Advanced Machine Learning with Social Dynamics Feature Engineering

### Overview
This project implements a comprehensive machine learning solution for predicting passenger survival on the Titanic. It features advanced feature engineering techniques, including a novel "Manipulative Potential Score" (MPS) that captures social dynamics and survival advantages based on passenger titles and class.

### Project Structure
```
titanic/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── model_training.py      # Model training and evaluation
│   └── visualization.py       # Plotting and visualization functions
├── output/                    # Generated outputs (plots, submissions)
├── models/                    # Saved trained models
├── train.csv                  # Training data
├── test.csv                   # Test data
├── gender_submission.csv      # Sample submission format
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Features

#### Novel Feature Engineering
- **Manipulative Potential Score (MPS)**: A unique feature that quantifies survival advantages based on social leverage
- **Title Extraction and Grouping**: Extracts titles from names and groups them into meaningful categories
- **Family Dynamics**: Features capturing family size and composition effects on survival
- **Age and Fare Binning**: Strategic categorization of continuous variables

#### Models
- **Random Forest**: Primary model for robust predictions
- **Gradient Boosting**: Secondary model for comparison
- **Ensemble**: Weighted combination of both models

### Installation

1. Clone the repository:
```bash
cd /d:/Titanic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Run the Complete Pipeline
```bash
python main.py
```

This will:
1. Load and explore the data
2. Engineer features
3. Preprocess and clean data
4. Train Random Forest and Gradient Boosting models
5. Evaluate models with cross-validation
6. Generate visualizations
7. Create submission files
8. Save trained models

#### Use in Jupyter Notebook
```python
# Import the modules
from src.feature_engineering import FeatureEngineer
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.visualization import Visualizer

# Initialize components
feature_engineer = FeatureEngineer()
preprocessor = DataPreprocessor()
trainer = ModelTrainer()
visualizer = Visualizer()

# Load your data
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Apply feature engineering
train_df_fe = feature_engineer.engineer_features(train_df)
test_df_fe = feature_engineer.engineer_features(test_df)

# Preprocess data
X_train, y_train, X_test, feature_cols = preprocessor.preprocess_data(train_df_fe, test_df_fe)

# Train models
results = trainer.train_all_models(X_train, y_train)

# Make predictions
predictions, probabilities = trainer.make_predictions(X_test, 'Random Forest')
```

### Output Files

The pipeline generates the following outputs:

#### Visualizations (in `output/`)
- `survival_analysis.png`: Comprehensive survival analysis by different features
- `model_comparison.png`: Model performance comparison
- `rf_feature_importance.png`: Random Forest feature importance
- `gb_feature_importance.png`: Gradient Boosting feature importance
- `prediction_summary.png`: Summary of predictions by model

#### Submission Files (in `output/`)
- `rf_submission.csv`: Random Forest predictions
- `gb_submission.csv`: Gradient Boosting predictions
- `ensemble_submission.csv`: Ensemble model predictions

#### Saved Models (in `models/`)
- `random_forest.pkl`: Trained Random Forest model
- `gradient_boosting.pkl`: Trained Gradient Boosting model
- `feature_columns.pkl`: List of feature columns used

### Key Insights

1. **Gender and Class**: Women and higher-class passengers had significantly higher survival rates
2. **Family Dynamics**: Medium-sized families (2-4 people) had better survival rates than solo travelers or large families
3. **Social Status**: Titles indicating higher social status (Noble, Military, Professional) showed interesting survival patterns
4. **Age**: Children had priority in lifeboats, as expected from historical accounts
5. **MPS Validation**: The Manipulative Potential Score successfully captured survival advantages from social leverage

### Model Performance

Typical performance metrics:
- Random Forest: ~83-84% accuracy (5-fold CV)
- Gradient Boosting: ~82-83% accuracy (5-fold CV)
- Models agree on ~90% of predictions

### Customization

You can customize the pipeline by modifying `src/config.py`:
- Adjust model hyperparameters
- Change feature engineering parameters
- Modify visualization settings

### Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

### Author
Titanic ML Project - Advanced survival prediction with social dynamics analysis 