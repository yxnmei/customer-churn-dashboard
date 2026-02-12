"""
Data loading utilities
"""

import pandas as pd
import yaml
from pathlib import Path


class DataLoader:
    """Handle data loading operations"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.preprocessed_dir = Path(self.config['data']['preprocessed_dir'])
        self.features_dir = Path(self.config['data']['features_dir'])
    
    def load_raw_data(self, filename="telco_churn.csv"):
        """Load raw data from CSV"""
        filepath = self.raw_dir / filename
        
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} rows, {df.shape[1]} columns")
        
        return df
    
    def save_preprocessed(self, df, filename):
        """Save preprocessed data"""
        filepath = self.preprocessed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✅ Saved to: {filepath}")
    
    def load_preprocessed(self, filename):
        """Load preprocessed data"""
        filepath = self.preprocessed_dir / filename
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} rows from {filepath}")
        return df
    
    def save_features(self, df, filename):
        """Save feature-engineered data"""
        filepath = self.features_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✅ Saved features to: {filepath}")
    
    def load_features(self, filename):
        """Load feature-engineered data"""
        filepath = self.features_dir / filename
        df = pd.read_csv(filepath)
        print(f"✅ Loaded features: {filepath}")
        return df


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    df = loader.load_raw_data()
    print(f"\nDataset preview:")
    print(df.head())