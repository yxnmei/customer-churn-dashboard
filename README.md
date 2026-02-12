# Customer Churn Prediction & What-If Analysis

A production-ready ML system for predicting customer churn with interactive dashboard and model interpretability.

## 🎯 Project Overview

Predicts which customers will churn and explains why, enabling proactive retention strategies.

**Key Features:**
- 79%+ ROC-AUC accuracy
- SHAP-based interpretability
- Interactive what-if simulator
- 293% estimated ROI

## 📁 Project Structure
```
churn_dashboard_project/
├── config/              # Configuration files
├── data/                # Data pipeline stages
├── notebooks/           # Exploratory analysis
├── src/                 # Source code
├── tests/               # Unit tests
└── dashboard/           # Streamlit app
```

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv churn
churn\Scripts\activate  # Windows
source churn/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Train model
python train.py

# Launch dashboard
streamlit run dashboard/app.py
```

## 📊 Results

- **ROC-AUC:** 79.3%
- **Precision:** 75.2%
- **Recall:** 70.6%
- **Est. Annual ROI:** $110,029

## 🔧 Tech Stack

- Python 3.13
- scikit-learn, XGBoost
- SHAP (interpretability)
- Streamlit (dashboard)
- Pandas, NumPy

## 👤 Author

Your Name - Data Science Portfolio Project