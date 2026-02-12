"""
Data preprocessing pipeline
Handles data cleaning and initial transformations
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Clean and preprocess raw data"""
    
    def __init__(self):
        self.cleaning_report = {}
    
    def fit_transform(self, df):
        """Run full preprocessing pipeline"""
        
        print("="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        df = df.copy()
        
        # 1. Handle TotalCharges data type issue
        df = self._fix_total_charges(df)
        
        # 2. Check for missing values
        df = self._handle_missing_values(df)
        
        # 3. Basic data validation
        self._validate_data(df)
        
        print("\n✅ Preprocessing complete!")
        return df
    
    def _fix_total_charges(self, df):
        """Fix TotalCharges being stored as object instead of numeric"""
        
        print("\n1️⃣ Fixing TotalCharges data type...")
        print(f"   Original type: {df['TotalCharges'].dtype}")
        
        # Convert to numeric (will create NaN for non-numeric values)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        missing_count = df['TotalCharges'].isnull().sum()
        print(f"   Found {missing_count} missing values")
        
        if missing_count > 0:
            # These are typically tenure=0 customers
            print(f"   Filling with MonthlyCharges (new customers)")
            df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = \
                df.loc[df['TotalCharges'].isnull(), 'MonthlyCharges']
        
        print(f"   New type: {df['TotalCharges'].dtype}")
        self.cleaning_report['total_charges_fixed'] = missing_count
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle any remaining missing values"""
        
        print("\n2️⃣ Checking for missing values...")
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        if total_missing > 0:
            print(f"   ⚠️  Found {total_missing} missing values:")
            print(missing[missing > 0])
            
            # Fill numeric with median, categorical with mode
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            
            print("   ✅ All missing values handled")
        else:
            print("   ✅ No missing values found")
        
        self.cleaning_report['missing_values_handled'] = total_missing
        return df
    
    def _validate_data(self, df):
        """Perform basic data validation"""
        
        print("\n3️⃣ Validating data...")
        
        # Check for duplicates
        duplicates = df.duplicated(subset='customerID').sum()
        if duplicates > 0:
            print(f"   ⚠️  Found {duplicates} duplicate customerIDs")
            df = df.drop_duplicates(subset='customerID')
        else:
            print(f"   ✅ No duplicate customerIDs")
        
        # Check data ranges
        assert df['tenure'].min() >= 0, "Negative tenure found!"
        assert df['MonthlyCharges'].min() > 0, "Zero or negative charges found!"
        
        print("   ✅ Data validation passed")
        
        self.cleaning_report['duplicates_removed'] = duplicates
        return df
    
    def get_report(self):
        """Return cleaning report"""
        return self.cleaning_report


if __name__ == "__main__":
    from src.utils.data_loader import DataLoader
    
    # Test preprocessing
    loader = DataLoader()
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df)
    
    # Save
    loader.save_preprocessed(df_clean, "telco_churn_clean.csv")
    
    print("\n" + "="*70)
    print("CLEANING REPORT:")
    print("="*70)
    for key, value in preprocessor.get_report().items():
        print(f"  {key}: {value}")