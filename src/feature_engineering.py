"""
Feature engineering module for Titanic survival prediction.
Contains all functions for creating and transforming features.
"""

import pandas as pd
import numpy as np
from .config import (
    TITLE_MAPPING, TITLE_SCORES, CLASS_MULTIPLIERS,
    AGE_BINS, AGE_LABELS, FARE_BINS, FARE_LABELS
)


class FeatureEngineer:
    """Class to handle all feature engineering operations."""
    
    def __init__(self):
        self.title_mapping = TITLE_MAPPING
        self.title_scores = TITLE_SCORES
        self.class_multipliers = CLASS_MULTIPLIERS
        
    @staticmethod
    def extract_title(name):
        """Extract title from passenger name."""
        try:
            title = name.split(',')[1].split('.')[0].strip()
            return title
        except:
            return 'Unknown'
    
    def create_title_groups(self, title):
        """Group rare titles into broader categories."""
        return self.title_mapping.get(title, 'Other')
    
    def calculate_mps(self, row):
        """
        Calculate Manipulative Potential Score based on title and class.
        This is a novel feature representing survival advantages from social leverage.
        """
        base_score = self.title_scores.get(row['TitleGroup'], 1)
        multiplier = self.class_multipliers.get(row['Pclass'], 1.0)
        return base_score * multiplier
    
    def create_family_features(self, df):
        """Create family-related features."""
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Categorize family size
        df['FamilySizeCategory'] = pd.cut(
            df['FamilySize'], 
            bins=[0, 1, 3, 5, 20], 
            labels=['Alone', 'Small', 'Medium', 'Large']
        )
        
        return df
    
    def create_age_fare_categories(self, df):
        """Create age and fare categorical features."""
        # Create age categories for non-null values
        df.loc[df['Age'].notna(), 'AgeCategory'] = pd.cut(
            df[df['Age'].notna()]['Age'], 
            bins=AGE_BINS, 
            labels=AGE_LABELS
        )
        
        # Create fare categories for non-null values
        df.loc[df['Fare'].notna(), 'FareCategory'] = pd.cut(
            df[df['Fare'].notna()]['Fare'], 
            bins=FARE_BINS, 
            labels=FARE_LABELS
        )
        
        return df
    
    def create_cabin_features(self, df):
        """Extract cabin-related features."""
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['CabinDeck'] = df['Cabin'].str[0].fillna('Unknown')
        return df
    
    def engineer_features(self, df):
        """Apply all feature engineering transformations."""
        df = df.copy()
        
        # Extract titles
        df['Title'] = df['Name'].apply(self.extract_title)
        df['TitleGroup'] = df['Title'].apply(self.create_title_groups)
        
        # Create family features
        df = self.create_family_features(df)
        
        # Calculate MPS
        df['MPS'] = df.apply(self.calculate_mps, axis=1)
        
        # Create age and fare categories
        df = self.create_age_fare_categories(df)
        
        # Create cabin features
        df = self.create_cabin_features(df)
        
        # Create binary sex feature
        df['IsFemale'] = (df['Sex'] == 'female').astype(int)
        
        return df
    
    def get_feature_names(self):
        """Return list of engineered feature names."""
        return [
            'Title', 'TitleGroup', 'FamilySize', 'IsAlone', 
            'FamilySizeCategory', 'MPS', 'AgeCategory', 
            'FareCategory', 'HasCabin', 'CabinDeck', 'IsFemale'
        ] 