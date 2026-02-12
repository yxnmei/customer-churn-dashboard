# Customer Churn Prediction & What-If Analysis

A production-ready ML system for predicting customer churn with interactive dashboard and model interpretability.

## 🎯 Project Overview

Predicts which customers will churn and explains why, enabling proactive retention strategies.

**Key Features:**
- 84% ROC-AUC accuracy
- SHAP-based interpretability
- Interactive what-if simulator
- Professional ML pipeline architecture

## 📁 Project Structure
```
churn_dashboard_project/
├── config/              # Configuration files
├── data/                # Data pipeline stages
│   ├── 01-raw/         # Original Kaggle dataset
│   ├── 02-preprocessed/ # Cleaned data
│   ├── 03-features/    # Engineered features
│   └── 04-predictions/ # Model outputs
├── src/                 # Source code
│   ├── pipelines/      # ML pipelines
│   ├── models/         # Model definitions
│   └── utils/          # Helper functions
├── models/             # Trained models & artifacts
├── dashboard/          # Streamlit app
└── notebooks/          # Exploratory analysis
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/customer-churn-dashboard.git
cd customer-churn-dashboard

# Create virtual environment
python -m venv churn
churn\Scripts\activate  # Windows
source churn/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Run Full Pipeline:**
```bash
# 1. Load and preprocess data
python -m src.pipelines.data_preprocessing

# 2. Engineer features
python -m src.pipelines.feature_engineering

# 3. Train models
python -m src.pipelines.model_training

# 4. Generate SHAP explanations
python -m src.pipelines.model_evaluation
```

**Launch Dashboard:**
```bash
streamlit run dashboard/app.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **84.09%** |
| **Accuracy** | 74.17% |
| **Recall** | 78.61% |
| **Precision** | 50.87% |
| **F1-Score** | 61.76% |

**Best Model:** Logistic Regression
- Chosen for optimal balance of performance and interpretability
- 78.6% capture rate of churning customers
- Fast inference for real-time dashboard

## 💡 Key Insights

**Top Churn Drivers (SHAP Analysis):**
1. **Internet Service** - Fiber optic customers show 42% churn rate
2. **Tenure** - First 12 months critical (48% churn vs 10% for 49+ months)
3. **Monthly Charges** - Price sensitivity significant predictor
4. **Contract Type** - Month-to-month: 43% churn vs 2-year: 3% churn
5. **Payment Method** - Electronic check users: 45% churn vs auto-pay: 16% churn

## 🎯 Dashboard Features

### 5 Interactive Pages:

1. **Overview** - Executive summary with key metrics
2. **Model Performance** - Detailed evaluation & confusion matrix
3. **Customer Explorer** - Browse customers by risk level
4. **What-If Simulator** ⭐ - Test retention strategies in real-time
5. **Feature Insights** - Deep dive into churn drivers

### What-If Simulator

Adjust customer attributes and see churn probability update instantly:
- Test contract upgrades
- Simulate service additions
- Calculate intervention ROI
- Generate personalized recommendations

## 🏗️ Technical Architecture

**ML Pipeline:**
- Data preprocessing with automated cleaning
- Feature engineering (8 domain-informed features)
- Model training with 4-algorithm comparison
- SHAP interpretability for explainability

**Models Trained:**
- Logistic Regression (Winner - 84% ROC-AUC)
- Random Forest (83% ROC-AUC)
- Gradient Boosting (84% ROC-AUC)
- XGBoost (84% ROC-AUC)

**Tech Stack:**
- **Python 3.13**
- **ML:** scikit-learn, XGBoost, imbalanced-learn
- **Interpretability:** SHAP
- **Dashboard:** Streamlit, Plotly
- **Data:** Pandas, NumPy

## 📈 Business Impact

**Estimated Annual Value:**
- Total customers at risk: 1,869
- Monthly revenue per customer: $64.76
- Annual revenue at risk: **$1.45M**

**With Model:**
- Churners identified: 78.6% capture rate
- Enables proactive retention campaigns
- ROI depends on intervention cost vs. retention success

## 🎓 Dataset

**Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Features:**
- 7,043 customers
- 21 original features
- 29 total features after engineering
- 26.54% churn rate

## 📝 Project Highlights

- ✅ Production-quality code structure
- ✅ Modular pipeline architecture
- ✅ SHAP explanations for every prediction
- ✅ Interactive what-if simulator
- ✅ Business-focused insights
- ✅ Deployment-ready

**Portfolio Value:**
- Demonstrates end-to-end ML workflow
- Shows production engineering skills
- Includes stakeholder-facing dashboard
- Proves ability to translate ML to business value

## 🚀 Deployment

Deployed on **Streamlit Community Cloud**

Live Demo: []

*Built as part of data science portfolio showcasing ML engineering, model interpretability, and business value creation.*