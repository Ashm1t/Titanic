"""
Data preprocessing module for Titanic survival prediction.
Handles data cleaning, missing value imputation, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import AGE_BINS, AGE_LABELS, FARE_BINS, FARE_LABELS


class DataPreprocessor:
    """Class to handle all data preprocessing operations."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Age: fill with median by class and sex
        for pclass in df['Pclass'].unique():
            for sex in df['Sex'].unique():
                mask = (df['Pclass'] == pclass) & (df['Sex'] == sex)
                median_age = df.loc[mask, 'Age'].median()
                if pd.notna(median_age):
                    df.loc[mask & df['Age'].isna(), 'Age'] = median_age
        
        # Fill any remaining Age NaN with overall median
        df['Age'].fillna(df['Age'].median(), inplace=True)
        
        # Fare: fill with median by class
        for pclass in df['Pclass'].unique():
            mask = df['Pclass'] == pclass
            median_fare = df.loc[mask, 'Fare'].median()
            if pd.notna(median_fare):
                df.loc[mask & df['Fare'].isna(), 'Fare'] = median_fare
        
        # Fill any remaining Fare NaN with overall median
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # Embarked: fill with mode
        if 'Embarked' in df.columns:
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        return df
    
    def update_categories_after_imputation(self, df):
        """Update age and fare categories after imputation."""
        # Re-create age categories
        df['AgeCategory'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS)
        df['AgeCategory'].fillna('Young_Adult', inplace=True)
        
        # Re-create fare categories
        df['FareCategory'] = pd.cut(df['Fare'], bins=FARE_BINS, labels=FARE_LABELS)
        df['FareCategory'].fillna('Low', inplace=True)
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols):
        """One-hot encode categorical features."""
        # Convert categorical variables to strings
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        # Handle Pclass as categorical
        if 'Pclass' in df_encoded.columns:
            pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
            df_encoded = pd.concat([df_encoded, pclass_dummies], axis=1)
            df_encoded.drop('Pclass', axis=1, inplace=True)
        
        return df_encoded
    
    def scale_features(self, X_train, X_test, numerical_cols):
        """Scale numerical features."""
        # Find columns that need scaling
        cols_to_scale = [col for col in X_train.columns if any(num_col in col for num_col in numerical_cols)]
        
        if cols_to_scale:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
            
            return X_train_scaled, X_test_scaled
        
        return X_train, X_test
    
    def preprocess_data(self, train_df, test_df):
        """Complete preprocessing pipeline."""
        # Features to drop
        features_to_drop = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Title']
        
        # Categorical columns to encode
        categorical_cols = ['Sex', 'Embarked', 'TitleGroup', 'FamilySizeCategory', 
                           'AgeCategory', 'FareCategory', 'CabinDeck']
        
        # Numerical columns to scale
        numerical_cols = ['Age', 'Fare', 'FamilySize', 'MPS', 'SibSp', 'Parch']
        
        # Combine datasets for consistent preprocessing
        all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # Handle missing values
        all_data = self.handle_missing_values(all_data)
        
        # Update categories after imputation
        all_data = self.update_categories_after_imputation(all_data)
        
        # Encode categorical features
        all_data_encoded = self.encode_categorical_features(all_data, categorical_cols)
        
        # Select features for modeling
        self.feature_cols = [col for col in all_data_encoded.columns if col not in features_to_drop]
        
        # Split back into train and test
        train_processed = all_data_encoded[:len(train_df)]
        test_processed = all_data_encoded[len(train_df):]
        
        X_train = train_processed[self.feature_cols]
        y_train = train_processed['Survived'].astype(int)
        X_test = test_processed[self.feature_cols]
        
        # Ensure all data is numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test, numerical_cols)
        
        return X_train, y_train, X_test, self.feature_cols 