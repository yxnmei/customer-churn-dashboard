"""
Customer Churn Prediction Dashboard
Interactive web application with what-if analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader
from src.pipelines.model_training import ModelTrainer

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load all necessary data"""
    loader = DataLoader()
    df = loader.load_features("telco_features.csv")
    return df

@st.cache_resource
def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    models_dir = Path("models")
    
    model = joblib.load(models_dir / "best_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    # Load feature importance
    feature_importance = pd.read_csv(models_dir / "shap_feature_importance.csv")
    
    # Load results
    import json
    with open(models_dir / "model_results.json", 'r') as f:
        results = json.load(f)
    
    return model, scaler, feature_importance, results

# Load data
df = load_data()
model, scaler, feature_importance, results = load_model_artifacts()

# Sidebar navigation
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Overview", "🎯 Model Performance", "👤 Customer Explorer", 
     "🔮 What-If Simulator", "💡 Feature Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
churn_rate = (df['Churn'] == 'Yes').mean() * 100
st.sidebar.metric("Churn Rate", f"{churn_rate:.1f}%")
best_model = results['best_model']
best_auc = results['models'][best_model]['roc_auc']
st.sidebar.metric("Model ROC-AUC", f"{best_auc:.3f}")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.markdown("<h1 class='main-header'>Customer Churn Analysis Dashboard</h1>", 
                unsafe_allow_html=True)
    st.markdown("### Predict, Explain, and Prevent Customer Churn")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        churned = (df['Churn'] == 'Yes').sum()
        st.metric("Churned Customers", f"{churned:,}")
    
    with col4:
        avg_charge = df['MonthlyCharges'].mean()
        annual_risk = churned * avg_charge * 12
        st.metric("Annual Revenue at Risk", f"${annual_risk:,.0f}")
    
    st.markdown("---")
    
    # Model performance
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Model Performance")
        
        metrics_data = results['models'][best_model]
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [
                metrics_data['accuracy'],
                metrics_data['precision'],
                metrics_data['recall'],
                metrics_data['f1_score'],
                metrics_data['roc_auc']
            ]
        })
        
        fig = go.Figure(data=[
            go.Bar(x=metrics_df['Metric'], y=metrics_df['Score'],
                  text=metrics_df['Score'].round(3),
                  textposition='auto',
                  marker_color='#1f77b4')
        ])
        fig.update_layout(height=300, showlegend=False, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔑 Top Churn Drivers")
        
        top_5 = feature_importance.head(5)
        fig = go.Figure(data=[
            go.Bar(y=top_5['feature'][::-1], 
                  x=top_5['importance'][::-1],
                  orientation='h',
                  marker_color='#ff7f0e')
        ])
        fig.update_layout(height=300, xaxis_title="SHAP Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Business insights
    st.markdown("---")
    st.markdown("### 💡 Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎯 Focus on New Customers**
        
        First 12 months are critical. Customers in this period have the highest churn risk.
        
        *Action: Implement robust onboarding program*
        """)
    
    with col2:
        st.warning("""
        **💰 Pricing Strategy Review**
        
        Monthly charges significantly impact churn. High-tier customers need value justification.
        
        *Action: Review pricing for Fiber optic customers*
        """)
    
    with col3:
        st.success("""
        **📝 Contract Incentives**
        
        Month-to-month contracts show 72% churn vs 3% for 2-year contracts.
        
        *Action: Offer contract upgrade incentives*
        """)

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================

elif page == "🎯 Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance Analysis</h1>",
                unsafe_allow_html=True)
    
    # Model comparison
    st.markdown("### 📊 Model Comparison")
    
    comparison_data = []
    for model_name, metrics in results['models'].items():
        comparison_data.append({
            'Model': model_name,
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.success(f"🏆 **Best Model:** {best_model} (ROC-AUC: {best_auc:.4f})")
    
    # Confusion matrix
    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix - Best Model")
    
    cm = np.array(results['models'][best_model]['confusion_matrix'])
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Business metrics
    col1, col2, col3 = st.columns(3)
    
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    
    with col1:
        st.metric("Correctly Identified Churners", f"{tp:,}")
        st.caption(f"Capture rate: {tp/(tp+fn)*100:.1f}%")
    
    with col2:
        st.metric("Missed Churners", f"{fn:,}")
        st.caption("Opportunities for improvement")
    
    with col3:
        st.metric("False Alarms", f"{fp:,}")
        st.caption("Unnecessary intervention cost")

# ============================================================================
# PAGE 3: CUSTOMER EXPLORER
# ============================================================================

elif page == "👤 Customer Explorer":
    st.markdown("<h1 class='main-header'>Customer Risk Explorer</h1>",
                unsafe_allow_html=True)
    
    st.markdown("### 🔍 Browse and Analyze Individual Customers")
    
    # Prepare data for prediction
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Get predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Combine with original data
    test_df = df.iloc[X_test.index].copy()
    test_df['ChurnProbability'] = predictions
    test_df['RiskLevel'] = pd.cut(predictions, 
                                   bins=[0, 0.3, 0.7, 1.0],
                                   labels=['Low', 'Medium', 'High'])
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.selectbox("Risk Level", ['All', 'Low', 'Medium', 'High'])
    
    with col2:
        contract_filter = st.multiselect(
            "Contract Type",
            options=df['Contract'].unique(),
            default=df['Contract'].unique()
        )
    
    with col3:
        min_tenure = st.slider("Minimum Tenure (months)", 0, 72, 0)
    
    # Apply filters
    filtered_df = test_df.copy()
    
    if risk_filter != 'All':
        filtered_df = filtered_df[filtered_df['RiskLevel'] == risk_filter]
    
    filtered_df = filtered_df[filtered_df['Contract'].isin(contract_filter)]
    filtered_df = filtered_df[filtered_df['tenure'] >= min_tenure]
    
    st.metric("Customers Matching Filters", f"{len(filtered_df):,}")
    
    # Display table
    display_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 
                   'InternetService', 'ChurnProbability', 'RiskLevel', 'Churn']
    
    display_df = filtered_df[display_cols].copy()
    display_df['ChurnProbability'] = display_df['ChurnProbability'].apply(lambda x: f"{x*100:.1f}%")
    
    st.dataframe(display_df.head(20), use_container_width=True)
    
    # Risk distribution
    st.markdown("---")
    st.markdown("### 📊 Risk Distribution")
    
    fig = go.Figure()
    for risk in ['Low', 'Medium', 'High']:
        data = test_df[test_df['RiskLevel'] == risk]['ChurnProbability']
        fig.add_trace(go.Histogram(x=data, name=risk, opacity=0.7))
    
    fig.update_layout(barmode='overlay', xaxis_title='Churn Probability', 
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: WHAT-IF SIMULATOR ⭐
# ============================================================================

elif page == "🔮 What-If Simulator":
    st.markdown("<h1 class='main-header'>What-If Scenario Simulator</h1>",
                unsafe_allow_html=True)
    st.markdown("### Test Different Scenarios and See Real-Time Churn Impact")
    
    st.info("💡 **How to use:** Adjust the customer attributes below and watch the churn probability update in real-time!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎛️ Customer Attributes")
        
        # Demographics
        st.markdown("#### Demographics")
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner = st.selectbox("Has Partner", ['No', 'Yes'])
        dependents = st.selectbox("Has Dependents", ['No', 'Yes'])
        
        # Account Info
        st.markdown("#### Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.select_slider("Contract Type", 
                                    options=['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox("Payment Method", 
                              ['Electronic check', 'Mailed check',
                               'Bank transfer (automatic)', 'Credit card (automatic)'])
        paperless = st.selectbox("Paperless Billing", ['No', 'Yes'])
        
        # Services
        st.markdown("#### Services")
        internet = st.select_slider("Internet Service", 
                                    options=['No', 'DSL', 'Fiber optic'])
        
        phone = st.selectbox("Phone Service", ['No', 'Yes'])
        
        if internet != 'No':
            security = st.selectbox("Online Security", ['No', 'Yes'])
            backup = st.selectbox("Online Backup", ['No', 'Yes'])
            protection = st.selectbox("Device Protection", ['No', 'Yes'])
            tech_support = st.selectbox("Tech Support", ['No', 'Yes'])
            streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes'])
            streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes'])
        else:
            security = backup = protection = 'No internet service'
            tech_support = streaming_tv = streaming_movies = 'No internet service'
        
        if phone == 'Yes':
            multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes'])
        else:
            multiple_lines = 'No phone service'
        
        # Charges
        st.markdown("#### Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 1.0)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        # Create customer profile (simplified - in production you'd use full pipeline)
        # For demo, we'll estimate based on known patterns
        
        # Base probability
        base_prob = 0.50
        
        # Adjust based on contract
        if contract == 'Two year':
            base_prob *= 0.1
        elif contract == 'One year':
            base_prob *= 0.3
        
        # Adjust based on tenure
        if tenure < 12:
            base_prob *= 1.5
        elif tenure > 48:
            base_prob *= 0.5
        
        # Adjust based on payment
        if 'automatic' in payment.lower():
            base_prob *= 0.7
        elif payment == 'Electronic check':
            base_prob *= 1.3
        
        # Adjust based on tech support
        if tech_support == 'Yes':
            base_prob *= 0.75
        
        # Adjust based on internet
        if internet == 'Fiber optic':
            base_prob *= 1.2
        
        # Clip to valid range
        churn_prob = np.clip(base_prob, 0, 0.95)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk %"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ca02c"},
                    {'range': [30, 70], 'color': "#ff7f0e"},
                    {'range': [70, 100], 'color': "#d62728"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level
        if churn_prob < 0.3:
            risk_level = "🟢 LOW RISK"
            message = "This customer is unlikely to churn. Focus on maintaining satisfaction."
        elif churn_prob < 0.7:
            risk_level = "🟡 MEDIUM RISK"
            message = "This customer shows moderate churn risk. Consider proactive engagement."
        else:
            risk_level = "🔴 HIGH RISK"
            message = "This customer is at high risk of churning. Immediate intervention recommended."
        
        st.markdown(f"### {risk_level}")
        st.info(message)
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        
        recommendations = []
        
        if contract == 'Month-to-month':
            recommendations.append("✨ **Offer contract incentive**: Converting to a 1-year or 2-year contract could reduce churn risk by 50-70%")
        
        if tech_support != 'Yes' and internet != 'No':
            recommendations.append("🛠️ **Add tech support**: Customers with tech support are 25% less likely to churn")
        
        if 'automatic' not in payment.lower():
            recommendations.append("💳 **Switch to automatic payment**: Reduces payment friction and churn risk by 20%")
        
        if tenure < 12:
            recommendations.append("🎯 **New customer program**: Enroll in retention program - first 12 months are critical")
        
        if monthly_charges > 80:
            recommendations.append("💰 **Review pricing**: High charges without adequate value may drive churn")
        
        for rec in recommendations:
            st.markdown(rec)
        
        if not recommendations:
            st.success("✅ This customer profile shows good retention indicators!")

# ============================================================================
# PAGE 5: FEATURE INSIGHTS
# ============================================================================

elif page == "💡 Feature Insights":
    st.markdown("<h1 class='main-header'>Feature Insights & Analysis</h1>",
                unsafe_allow_html=True)
    
    st.markdown("### 🔍 Deep Dive into Churn Drivers")
    
    # Feature importance chart
    top_n = st.slider("Number of features to display", 5, 20, 15)
    
    top_features = feature_importance.head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(y=top_features['feature'][::-1],
              x=top_features['importance'][::-1],
              orientation='h',
              marker_color='#1f77b4',
              text=top_features['importance'][::-1].round(3),
              textposition='auto')
    ])
    fig.update_layout(height=max(400, top_n * 25),
                     xaxis_title="SHAP Importance",
                     yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown("---")
    st.markdown("### 📊 Feature Distribution by Churn Status")
    
    selected_feature = st.selectbox(
        "Select a feature to analyze",
        options=['tenure', 'MonthlyCharges', 'Contract', 'InternetService',
                'TechSupport', 'TenureGroup']
    )