"""
Configuration file for Titanic Survival Prediction project.
Contains all parameters, constants, and settings.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Data file names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "gender_submission.csv"

# Random seed for reproducibility
RANDOM_SEED = 42

# Feature engineering parameters
AGE_BINS = [0, 12, 18, 35, 55, 80]
AGE_LABELS = ['Child', 'Teen', 'Young_Adult', 'Middle_Aged', 'Senior']
FARE_BINS = [-0.001, 10, 30, 100, 600]
FARE_LABELS = ['Low', 'Medium', 'High', 'Very_High']

# Title mappings
TITLE_MAPPING = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Professional',
    'Rev': 'Professional',
    'Col': 'Military',
    'Major': 'Military',
    'Capt': 'Military',
    'Sir': 'Noble',
    'Lady': 'Noble',
    'Don': 'Noble',
    'Dona': 'Noble',
    'Countess': 'Noble',
    'Jonkheer': 'Noble',
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs'
}

# MPS (Manipulative Potential Score) parameters
TITLE_SCORES = {
    'Mr': 2,
    'Mrs': 1,
    'Miss': 1,
    'Master': 0,
    'Professional': 3,
    'Military': 4,
    'Noble': 5,
    'Other': 1
}

CLASS_MULTIPLIERS = {1: 2.0, 2: 1.5, 3: 1.0}

# Model parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

GB_PARAMS = {
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 4,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'subsample': 0.8,
    'random_state': RANDOM_SEED
}

# Cross-validation parameters
CV_FOLDS = 5
TEST_SIZE = 0.2

# Visualization parameters
FIGURE_SIZE = (15, 10)
STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'husl' 