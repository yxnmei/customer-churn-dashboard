"""
Model evaluation and interpretability with SHAP
Explains WHY the model makes predictions
"""

import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt


class ModelEvaluator:
    """Evaluate model with SHAP interpretability"""
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.shap_values = None
        
    def load_model(self, model_name="best_model.pkl"):
        """Load trained model"""
        
        print("="*70)
        print("LOADING MODEL FOR EVALUATION")
        print("="*70)
        
        models_dir = Path("models")
        model_path = models_dir / model_name
        
        print(f"\nLoading model from: {model_path}")
        self.model = joblib.load(model_path)
        print(f"✅ Loaded: {type(self.model).__name__}")
        
        return self.model
    
    def create_shap_explainer(self, X_train, model_type="logistic"):
        """Create SHAP explainer"""
        
        print("\n" + "="*70)
        print("CREATING SHAP EXPLAINER")
        print("="*70)
        
        print(f"\nModel type: {model_type}")
        
        if model_type.lower() in ['logistic', 'linear']:
            # Fast exact computation for linear models
            self.explainer = shap.LinearExplainer(self.model, X_train)
            print("✅ Using LinearExplainer (exact, fast)")
            
        elif model_type.lower() in ['tree', 'forest', 'boost', 'xgboost']:
            # Fast exact computation for tree models
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ Using TreeExplainer (exact, fast)")
            
        else:
            # Model-agnostic explainer
            background = shap.sample(X_train, 100)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background
            )
            print("✅ Using KernelExplainer (model-agnostic)")
        
        return self.explainer
    
    def compute_shap_values(self, X_test):
        """Compute SHAP values for test set"""
        
        print("\n" + "="*70)
        print("COMPUTING SHAP VALUES")
        print("="*70)
        
        print(f"\nComputing SHAP values for {len(X_test):,} samples...")
        
        self.shap_values = self.explainer.shap_values(X_test)
        
        # For binary classification, some explainers return list
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class
        
        print(f"✅ SHAP values computed: {self.shap_values.shape}")
        
        return self.shap_values
    
    def analyze_global_importance(self, X_test, top_n=15):
        """Analyze global feature importance"""
        
        print("\n" + "="*70)
        print("GLOBAL FEATURE IMPORTANCE")
        print("="*70)
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top {top_n} Most Important Features:")
        print(feature_importance.head(top_n).to_string(index=False))
        
        # Save to file
        models_dir = Path("models")
        feature_importance.to_csv(
            models_dir / "shap_feature_importance.csv", 
            index=False
        )
        print("\n✅ Saved: models/shap_feature_importance.csv")
        
        return feature_importance
    
    def analyze_feature_direction(self, X_test, top_n=10):
        """Analyze feature impact direction"""
        
        print("\n" + "="*70)
        print("FEATURE IMPACT DIRECTION")
        print("="*70)
        
        feature_direction = pd.DataFrame({
            'feature': X_test.columns,
            'mean_shap': self.shap_values.mean(axis=0),
            'abs_importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('abs_importance', ascending=False)
        
        print(f"\n📊 Top {top_n} Features by Direction:")
        for idx, row in feature_direction.head(top_n).iterrows():
            feature = row['feature']
            direction = row['mean_shap']
            
            if direction > 0:
                emoji = "📈"
                impact = "INCREASES"
            else:
                emoji = "📉"
                impact = "DECREASES"
            
            print(f"   {emoji} {feature:<25s} {impact} churn ({direction:+.4f})")
        
        return feature_direction
    
    def get_business_insights(self, feature_importance, top_n=5):
        """Generate business-friendly insights"""
        
        print("\n" + "="*70)
        print("BUSINESS INSIGHTS")
        print("="*70)
        
        print(f"\n💡 Top {top_n} Actionable Insights:\n")
        
        top_features = feature_importance.head(top_n)
        
        # Map features to business actions
        action_map = {
            'Contract': 'Offer incentives for longer contracts',
            'tenure': 'Focus retention on new customers (0-12 months)',
            'TenureGroup': 'Invest in first-year customer experience',
            'MonthlyCharges': 'Review pricing strategy for high-tier customers',
            'InternetService': 'Investigate Fiber optic pricing/experience',
            'TechSupport': 'Promote tech support as retention tool',
            'OnlineSecurity': 'Bundle security services in offers',
            'IsStableCustomer': 'Incentivize automatic payment adoption',
            'HasSupportServices': 'Cross-sell support services',
            'PaymentMethod': 'Reduce friction in payment process',
            'Payment_Electronic check': 'Migrate electronic check users to auto-pay'
        }
        
        for i, (idx, row) in enumerate(top_features.iterrows(), 1):
            feature = row['feature']
            importance = row['importance']
            
            action = action_map.get(feature, f"Analyze {feature} impact on retention")
            
            print(f"{i}. {feature}")
            print(f"   Impact: {importance:.4f}")
            print(f"   Action: {action}\n")
        
        return action_map
    
    def save_explainer(self):
        """Save SHAP explainer for dashboard use"""
        
        print("\n" + "="*70)
        print("SAVING SHAP ARTIFACTS")
        print("="*70)
        
        models_dir = Path("models")
        
        # Save explainer
        joblib.dump(self.explainer, models_dir / "shap_explainer.pkl")
        print("✅ Saved SHAP explainer")
        
        # Save example SHAP values (for dashboard preview)
        np.save(models_dir / "shap_values_sample.npy", self.shap_values[:100])
        print("✅ Saved sample SHAP values")


if __name__ == "__main__":
    from src.utils.data_loader import DataLoader
    from src.pipelines.model_training import ModelTrainer
    
    # Load feature data
    loader = DataLoader()
    df = loader.load_features("telco_features.csv")
    
    # Prepare data (same as training)
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load best model
    evaluator.load_model("best_model.pkl")
    
    # Create SHAP explainer
    evaluator.create_shap_explainer(X_train, model_type="logistic")
    
    # Compute SHAP values
    evaluator.compute_shap_values(X_test)
    
    # Analyze importance
    feature_importance = evaluator.analyze_global_importance(X_test, top_n=15)
    
    # Analyze direction
    evaluator.analyze_feature_direction(X_test, top_n=10)
    
    # Get business insights
    evaluator.get_business_insights(feature_importance, top_n=5)
    
    # Save everything
    evaluator.save_explainer()
    
    print("\n" + "="*70)
    print("✅ SHAP ANALYSIS COMPLETE!")
    print("="*70)