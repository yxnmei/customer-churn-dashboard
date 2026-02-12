"""
Model training pipeline
Trains multiple models, compares performance, saves best model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib
import yaml
from pathlib import Path
import json


class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.results = {}
        self.scaler = None
        self.label_encoders = {}
        
    def prepare_data(self, df):
        """Encode and split data"""
        
        print("="*70)
        print("DATA PREPARATION FOR MODELING")
        print("="*70)
        
        df = df.copy()
        
        # Encode categorical variables
        print("\n1️⃣ Encoding categorical variables...")
        df = self._encode_features(df)
        
        # Separate features and target
        print("\n2️⃣ Separating features and target...")
        X = df.drop(['customerID', 'Churn'], axis=1)
        y = (df['Churn'] == 'Yes').astype(int)
        
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {len(X):,}")
        print(f"   Churn rate: {y.mean()*100:.2f}%")
        
        # Train-test split
        print("\n3️⃣ Splitting data...")
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale numerical features
        print("\n4️⃣ Scaling features...")
        X_train, X_test = self._scale_features(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
    
    def _encode_features(self, df):
        """Encode all categorical variables"""
        
        # Binary Yes/No columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = (df[col] == 'Yes').astype(int)
        
        # Gender
        if 'gender' in df.columns:
            df['gender'] = (df['gender'] == 'Male').astype(int)
        
        # Ordinal features
        ordinal_maps = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'TenureGroup': {'New': 0, 'Mid': 1, 'Established': 2, 'Loyal': 3},
            'MonthlyChargeTier': {'Budget': 0, 'Standard': 1, 'Premium': 2, 'Elite': 3}
        }
        
        for col, mapping in ordinal_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # One-hot encode PaymentMethod
        if 'PaymentMethod' in df.columns:
            df = pd.get_dummies(df, columns=['PaymentMethod'], prefix='Payment', drop_first=True)
        
        # Multi-level service features
        multilevel_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in multilevel_cols:
            if col in df.columns:
                df[col] = df[col].map({
                    'No': 0,
                    'Yes': 1,
                    'No phone service': -1,
                    'No internet service': -1
                })
        
        print(f"   Encoded features: {df.shape[1]}")
        
        return df
    
    def _scale_features(self, X_train, X_test):
        """Scale numerical features"""
        
        # Only scale truly continuous features
        features_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'PricePerService']
        features_to_scale = [f for f in features_to_scale if f in X_train.columns]
        
        self.scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[features_to_scale] = self.scaler.fit_transform(X_train[features_to_scale])
        X_test_scaled[features_to_scale] = self.scaler.transform(X_test[features_to_scale])
        
        print(f"   Scaled {len(features_to_scale)} features")
        
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        random_state = self.config['model']['random_state']
        
        # 1. Logistic Regression
        print("\n🔵 Training Logistic Regression...")
        lr = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        print("   ✅ Complete")
        
        # 2. Random Forest
        print("\n🌲 Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print("   ✅ Complete")
        
        # 3. Gradient Boosting
        print("\n⚡ Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        print("   ✅ Complete")
        
        # 4. XGBoost
        print("\n🚀 Training XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        print("   ✅ Complete")
        
        print(f"\n✅ Trained {len(self.models)} models!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            r = self.results[name]
            print(f"   Accuracy:  {r['accuracy']:.4f}")
            print(f"   Precision: {r['precision']:.4f}")
            print(f"   Recall:    {r['recall']:.4f}")
            print(f"   F1-Score:  {r['f1_score']:.4f}")
            print(f"   ROC-AUC:   {r['roc_auc']:.4f}")
        
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   ROC-AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        
        return best_model_name
    
    def save_artifacts(self, best_model_name, X_train):
        """Save models and related artifacts"""
        
        print("\n" + "="*70)
        print("SAVING ARTIFACTS")
        print("="*70)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save best model
        best_model = self.models[best_model_name]
        joblib.dump(best_model, models_dir / "best_model.pkl")
        print(f"✅ Saved best model: {best_model_name}")
        
        # Save all models
        for name, model in self.models.items():
            safe_name = name.replace(' ', '_').lower()
            joblib.dump(model, models_dir / f"{safe_name}.pkl")
        print(f"✅ Saved all {len(self.models)} models")
        
        # Save scaler
        joblib.dump(self.scaler, models_dir / "scaler.pkl")
        print("✅ Saved scaler")
        
        # Save feature names
        feature_names = X_train.columns.tolist()
        with open(models_dir / "feature_names.txt", 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print("✅ Saved feature names")
        
        # Save results
        results_summary = {
            'best_model': best_model_name,
            'models': self.results
        }
        
        with open(models_dir / "model_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        print("✅ Saved results summary")


if __name__ == "__main__":
    from src.utils.data_loader import DataLoader
    
    # Load feature-engineered data
    loader = DataLoader()
    df = loader.load_features("telco_features.csv")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train models
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    best_model_name = trainer.evaluate_models(X_test, y_test)
    
    # Save everything
    trainer.save_artifacts(best_model_name, X_train)
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*70)