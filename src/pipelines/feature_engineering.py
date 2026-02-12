"""
Feature engineering pipeline
Creates domain-informed features for churn prediction
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class FeatureEngineer:
    """Engineer features based on business insights"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_report = {}
    
    def fit_transform(self, df):
        """Run full feature engineering pipeline"""
        
        print("="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        df = df.copy()
        original_features = df.shape[1]
        
        # Create new features
        df = self._create_tenure_groups(df)
        df = self._create_service_count(df)
        df = self._create_support_flag(df)
        df = self._create_stability_score(df)
        df = self._create_charge_tier(df)
        df = self._create_price_per_service(df)
        df = self._create_vulnerable_segments(df)
        
        new_features = df.shape[1] - original_features
        print(f"\n✅ Created {new_features} new features!")
        print(f"   Total features: {df.shape[1]}")
        
        self.feature_report['original_features'] = original_features
        self.feature_report['engineered_features'] = new_features
        self.feature_report['total_features'] = df.shape[1]
        
        return df
    
    def _create_tenure_groups(self, df):
        """Create customer lifecycle stages"""
        
        print("\n📊 Creating TenureGroup (lifecycle stages)...")
        
        bins = self.config['features']['tenure_bins']
        labels = self.config['features']['tenure_labels']
        
        df['TenureGroup'] = pd.cut(
            df['tenure'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        print(f"   Distribution:")
        print(df['TenureGroup'].value_counts().to_string())
        
        return df
    
    def _create_service_count(self, df):
        """Count total services subscribed"""
        
        print("\n📊 Creating ServiceCount...")
        
        service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies'
        ]
        
        df['ServiceCount'] = 0
        for col in service_cols:
            df['ServiceCount'] += (df[col] == 'Yes').astype(int)
        
        print(f"   Range: {df['ServiceCount'].min()} to {df['ServiceCount'].max()}")
        print(f"   Average: {df['ServiceCount'].mean():.2f} services")
        
        return df
    
    def _create_support_flag(self, df):
        """Flag for protective support services"""
        
        print("\n📊 Creating HasSupportServices...")
        
        df['HasSupportServices'] = (
            (df['OnlineSecurity'] == 'Yes') |
            (df['TechSupport'] == 'Yes') |
            (df['DeviceProtection'] == 'Yes')
        ).astype(int)
        
        count = df['HasSupportServices'].sum()
        print(f"   {count:,} customers have support services")
        
        return df
    
    def _create_stability_score(self, df):
        """Identify stable customers (long contract + auto-pay)"""
        
        print("\n📊 Creating IsStableCustomer...")
        
        df['IsStableCustomer'] = (
            (df['Contract'] != 'Month-to-month') &
            (df['PaymentMethod'].str.contains('automatic'))
        ).astype(int)
        
        count = df['IsStableCustomer'].sum()
        print(f"   {count:,} stable customers identified")
        
        return df
    
    def _create_charge_tier(self, df):
        """Categorize by monthly charges"""
        
        print("\n📊 Creating MonthlyChargeTier...")
        
        q = self.config['features']['charge_quantiles']
        df['MonthlyChargeTier'] = pd.qcut(
            df['MonthlyCharges'],
            q=q,
            labels=['Budget', 'Standard', 'Premium', 'Elite']
        )
        
        print(f"   Distribution:")
        print(df['MonthlyChargeTier'].value_counts().to_string())
        
        return df
    
    def _create_price_per_service(self, df):
        """Calculate value perception metric"""
        
        print("\n📊 Creating PricePerService...")
        
        df['PricePerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
        
        # Handle any edge cases
        if df['PricePerService'].isnull().any():
            df['PricePerService'].fillna(df['PricePerService'].median(), inplace=True)
        
        print(f"   Average: ${df['PricePerService'].mean():.2f} per service")
        
        return df
    
    def _create_vulnerable_segments(self, df):
        """Identify vulnerable customer segments"""
        
        print("\n📊 Creating vulnerability flags...")
        
        # Senior without support
        df['SeniorWithoutSupport'] = (
            (df['SeniorCitizen'] == 1) &
            (df['Partner'] == 'No') &
            (df['Dependents'] == 'No') &
            (df['TechSupport'] == 'No')
        ).astype(int)
        
        senior_count = df['SeniorWithoutSupport'].sum()
        print(f"   {senior_count:,} vulnerable seniors")
        
        # New customer with high charges (price shock)
        high_charge = df['MonthlyCharges'].quantile(0.75)
        df['NewCustomerHighCharge'] = (
            (df['tenure'] < 6) &
            (df['MonthlyCharges'] > high_charge)
        ).astype(int)
        
        shock_count = df['NewCustomerHighCharge'].sum()
        print(f"   {shock_count:,} new customers with price shock")
        
        return df
    
    def get_report(self):
        """Return feature engineering report"""
        return self.feature_report


if __name__ == "__main__":
    from src.utils.data_loader import DataLoader
    
    # Load cleaned data
    loader = DataLoader()
    df = loader.load_preprocessed("telco_churn_clean.csv")
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.fit_transform(df)
    
    # Save
    loader.save_features(df_features, "telco_features.csv")
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING REPORT:")
    print("="*70)
    for key, value in engineer.get_report().items():
        print(f"  {key}: {value}")