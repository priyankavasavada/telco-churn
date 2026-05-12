import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📊")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown, .stText, span {
    color: #e0e0e0 !important;
}

.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 48px rgba(0,0,0,0.4);
}

.input-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
    backdrop-filter: blur(10px);
}

.divider {
    background: linear-gradient(90deg, transparent, rgba(138, 43, 226, 0.5), transparent);
    height: 2px;
    border: none;
    margin: 24px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}

.result-card-success {
    background: linear-gradient(135deg, rgba(0, 230, 118, 0.15) 0%, rgba(0, 200, 83, 0.05) 100%);
    border: 1px solid rgba(0, 230, 118, 0.3);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-card-danger {
    background: linear-gradient(135deg, rgba(255, 82, 82, 0.15) 0%, rgba(200, 50, 50, 0.05) 100%);
    border: 1px solid rgba(255, 82, 82, 0.3);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    color: #b0b0b0 !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.stSelectbox, .stSlider, .stNumberInput {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: none !important;
    color: white !important;
}

.stSlider > div > div {
    color: #667eea !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
}

.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #a0a0ff !important;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stDataFrame {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}

.stDataFrame [data-testid="StyledDataFrameDataCell"] {
    color: #e0e0e0 !important;
}

.stExpander {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #667eea, #764ba2);
    border-radius: 4px;
}

.insight-box {
    background: rgba(102, 126, 234, 0.1);
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    import os, subprocess, sys
    if not os.path.exists('model.pkl'):
        with st.spinner("Training model for the first time... This may take a minute."):
            subprocess.run([sys.executable, 'train.py'], check=True)
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('telco_churn.csv')

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
feature_names = artifacts['feature_names']
model_report = artifacts.get('model_report', {})
best_model_name = artifacts.get('best_model_name', 'Model')
final_auc = artifacts.get('final_test_auc', 0)
final_acc = artifacts.get('final_test_accuracy', 0)

st.markdown("""
<div style="text-align: center; padding: 32px 0 16px 0;">
    <h1 style="font-size: 48px; background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text; margin-bottom: 4px;">
        Telco Customer Churn Predictor
    </h1>
    <p style="color: #a0a0a0; font-size: 16px; letter-spacing: 2px; text-transform: uppercase;">
        Predict · Analyze · Retain
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Explore Data", "🤖 Model Performance"])

with tab1:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <div>
            <h2 style="margin: 0;">Customer Details</h2>
            <p style="color: #a0a0a0; margin: 4px 0 0 0;">
                Fill in the customer information below to predict churn probability
            </p>
        </div>
        <div style="background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
                    border-radius: 12px; padding: 8px 20px; text-align: center;">
            <p style="margin: 0; font-size: 12px; color: #a0a0ff;">Best Model</p>
            <p style="margin: 0; font-weight: 700; color: white;">{}</p>
        </div>
    </div>
    """.format(best_model_name), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="section-header">👤 Demographics</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            Partner = st.selectbox("Has Partner", ["Yes", "No"])
            Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)

        with col2:
            st.markdown('<p class="section-header">📞 Phone Services</p>', unsafe_allow_html=True)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

            st.markdown('<p class="section-header" style="margin-top: 16px;">🌐 Internet Services</p>', unsafe_allow_html=True)
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown('<p class="section-header">📄 Account Info</p>', unsafe_allow_html=True)
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=1.0)

        st.markdown('</div>', unsafe_allow_html=True)

    TotalCharges = round(tenure * MonthlyCharges * np.random.uniform(0.95, 1.05), 2)

    input_data = {
        'gender': gender, 'SeniorCitizen': int(SeniorCitizen == 'Yes'),
        'Partner': Partner, 'Dependents': Dependents, 'tenure': tenure,
        'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
        'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup, 'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies, 'Contract': Contract,
        'PaperlessBilling': PaperlessBilling, 'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges,
    }

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔮 Predict Churn", type="primary", width='stretch')

    if predict_clicked:
        try:
            df = pd.DataFrame([input_data])

            for col in label_encoders:
                if col in df.columns:
                    df[col] = label_encoders[col].transform(df[col])

            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df[num_cols] = df[num_cols].astype(float)
            df[num_cols] = scaler.transform(df[num_cols])
            df = df[feature_names]

            prob = model.predict_proba(df)[0, 1]
            pred = model.predict(df)[0]
        except Exception as e:
            st.error(f"**Prediction Error:** {type(e).__name__}")
            st.code(str(e))
            st.stop()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col_res1, col_res2 = st.columns([1, 1.5])

        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-card-danger">
                    <div style="font-size: 64px; margin-bottom: 8px;">⚠️</div>
                    <h2 style="color: #ff5252 !important; margin: 0;">{prob:.1%} Churn Risk</h2>
                    <p style="color: #ff8a80; font-size: 18px; margin: 8px 0 0 0;">
                        This customer is <strong>likely to churn</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-success">
                    <div style="font-size: 64px; margin-bottom: 8px;">✅</div>
                    <h2 style="color: #00e676 !important; margin: 0;">{prob:.1%} Churn Risk</h2>
                    <p style="color: #69f0ae; font-size: 18px; margin: 8px 0 0 0;">
                        This customer is <strong>likely to stay</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin-top: 16px;">
                <p style="color: #a0a0a0; margin: 0 0 8px 0; font-size: 14px;">Prediction Confidence</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #00e676; font-weight: 600;">Stay {1-prob:.0%}</span>
                    <span style="color: #ff5252; font-weight: 600;">Churn {prob:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.subheader("Top Risk Factors")

            input_row = df.iloc[0:1]
            baseline = pd.DataFrame(0, index=[0], columns=feature_names)
            baseline_proba = model.predict_proba(baseline)[0, 1]
            contributions = []

            for i, col in enumerate(feature_names):
                perturbed = baseline.copy()
                perturbed.iloc[0, i] = input_row.iloc[0, i]
                perturbed_proba = model.predict_proba(perturbed)[0, 1]
                contributions.append((col, perturbed_proba - baseline_proba))

            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top = contributions[:10]

            labels = [c[0] for c in top]
            values = [c[1] * 100 for c in top]
            colors = ['#ff5252' if v > 0 else '#00e676' for v in values]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
            ax.axvline(0, color=(1.0, 1.0, 1.0, 0.3), linestyle='-', linewidth=0.5)
            ax.set_xlabel('Impact on Churn Probability (%)', color='#a0a0a0', fontsize=12)
            ax.set_title('Feature Contributions to Churn Risk', color='white', fontsize=14, fontweight='bold')
            ax.tick_params(colors='#e0e0e0', labelsize=11)

            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))

            pad = max(abs(min(values)), abs(max(values))) * 0.1
            for bar, val in zip(bars, values[::-1]):
                if val > 0:
                    ax.text(val + pad, bar.get_y() + bar.get_height()/2,
                            f'+{val:.1f}%', va='center', ha='left',
                            color='#ff5252', fontweight='bold', fontsize=10)
                else:
                    ax.text(val - pad, bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', va='center', ha='right',
                            color='#00e676', fontweight='bold', fontsize=10)

            margin = max(abs(min(values)), abs(max(values))) * 0.25 if values else 5
            ax.set_xlim(min(values) - margin, max(values) + margin)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; gap: 16px; flex-wrap: wrap;">
            <div class="insight-box" style="flex: 1;">
                <p style="margin: 0; font-weight: 600; color: white !important;">💡 Tip</p>
                <p style="margin: 4px 0 0 0; font-size: 14px;">
                    High churn risk customers should be targeted with retention offers,
                    especially those with month-to-month contracts and fiber optic internet.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    df = load_data()

    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <div>
            <h2 style="margin: 0;">Dataset Explorer</h2>
            <p style="color: #a0a0a0; margin: 4px 0 0 0;">
                Explore the Telco customer churn dataset
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    churners = df['Churn'].value_counts().get('Yes', 0)
    non_churners = df['Churn'].value_counts().get('No', 0)
    churn_pct = churners / total * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Total Customers</p>
            <p class="metric-value" style="background: linear-gradient(135deg, #667eea, #764ba2);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text; margin: 4px 0;">{total:,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Churned</p>
            <p class="metric-value" style="color: #ff5252 !important; margin: 4px 0;">{churners:,}</p>
            <p style="color: #ff8a80; margin: 0; font-size: 14px;">{churn_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Retained</p>
            <p class="metric-value" style="color: #00e676 !important; margin: 4px 0;">{non_churners:,}</p>
            <p style="color: #69f0ae; margin: 0; font-size: 14px;">{100-churn_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        avg_tenure = df['tenure'].mean()
        avg_charges = df['MonthlyCharges'].mean()
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Avg Tenure / Charges</p>
            <p class="metric-value" style="color: #f093fb !important; margin: 4px 0;">{avg_tenure:.0f}m</p>
            <p style="color: #e0a0e0; margin: 0; font-size: 14px;">${avg_charges:.0f}/mo</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.expander("📋 View Raw Data", expanded=False):
        st.dataframe(df.head(100), width='stretch')
        st.markdown(f"""
        <p style="color: #a0a0a0; font-size: 13px;">
            Showing 100 of {len(df):,} rows · {len(df.columns)} columns
        </p>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3>📈 Churn Analysis</h3>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        colors_churn = ['#00e676', '#ff5252']
        churn_counts = df['Churn'].value_counts()
        wedges, texts, autotexts = ax.pie(
            churn_counts.values,
            labels=churn_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors_churn,
            explode=(0.05, 0.05),
            shadow=False,
            textprops={'color': 'white', 'fontweight': 'bold', 'fontsize': 12}
        )
        ax.set_title('Churn Distribution', color='white', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)

    with col_c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        churn_0 = df[df['Churn'] == 'No']['tenure']
        churn_1 = df[df['Churn'] == 'Yes']['tenure']

        ax.hist(churn_0, bins=30, alpha=0.7, label='Retained', color='#00e676', edgecolor='none')
        ax.hist(churn_1, bins=30, alpha=0.7, label='Churned', color='#ff5252', edgecolor='none')
        ax.set_xlabel('Tenure (months)', color='#a0a0a0', fontsize=12)
        ax.set_ylabel('Count', color='#a0a0a0', fontsize=12)
        ax.set_title('Tenure Distribution by Churn', color='white', fontsize=14, fontweight='bold')
        ax.legend(facecolor='none', labelcolor=['#00e676', '#ff5252'], fontsize=11)
        ax.tick_params(colors='#e0e0e0', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.1))
        st.pyplot(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3>🔍 Key Insights by Feature</h3>', unsafe_allow_html=True)

    cat_insights = ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen']
    insight_cols = st.columns(2)

    for idx, col_name in enumerate(cat_insights):
        with insight_cols[idx % 2]:
            n_cats = df[col_name].nunique()
            w = max(5.0, n_cats * 1.5)
            fig, ax = plt.subplots(figsize=(w, 4.0))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            ct = pd.crosstab(df[col_name], df['Churn'], normalize='index') * 100
            ct.plot(kind='bar', ax=ax, color=['#00e676', '#ff5252'], width=0.65, legend=False)
            ax.set_title(f'Churn Rate by {col_name}', color='white', fontsize=13, fontweight='bold', pad=12)
            ax.set_ylabel('% of Customers', color='#a0a0a0', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylim(0, 100)
            plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
            ax.tick_params(colors='#e0e0e0', labelsize=9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))
            ax.legend(['Retained', 'Churned'], facecolor='none', labelcolor=['#00e676', '#ff5252'],
                      fontsize=9, loc='upper right', framealpha=0)
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(16, 13))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    df_corr = df.copy()
    for col in df_corr.select_dtypes(include='object').columns:
        df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, cbar_kws={'label': ''}, square=False,
                annot_kws={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
                linewidths=0.5, linecolor=(1.0, 1.0, 1.0, 0.15))
    ax.set_title('Feature Correlation Matrix', color='white', fontsize=16, fontweight='bold', pad=24)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.tick_params(colors='#e0e0e0', labelsize=10)
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <div>
            <h2 style="margin: 0;">Model Performance</h2>
            <p style="color: #a0a0a0; margin: 4px 0 0 0;">
                Compare model architectures and see detailed metrics
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_bm1, col_bm2, col_bm3 = st.columns(3)
    with col_bm1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Best Model</p>
            <p class="metric-value" style="color: #f093fb !important; margin: 8px 0; font-size: 28px;">
                {best_model_name}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_bm2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Test AUC</p>
            <p class="metric-value" style="color: #667eea !important; margin: 8px 0; font-size: 28px;">
                {final_auc:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_bm3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="color: #a0a0a0; margin: 0; font-size: 14px;">Test Accuracy</p>
            <p class="metric-value" style="color: #00e676 !important; margin: 8px 0; font-size: 28px;">
                {final_acc:.2%}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if model_report:
        st.markdown('<h3>📊 Model Comparison</h3>', unsafe_allow_html=True)

        comparison_data = []
        for name, report in model_report.items():
            if name == 'feature_importances':
                continue
            info = report.get('best_params', {}) if isinstance(report, dict) else {}
            if isinstance(report, dict) and 'cv_auc_mean' in report:
                comparison_data.append({
                    'Model': name,
                    'CV AUC': f"{report['cv_auc_mean']:.4f} (±{report['cv_auc_std']:.4f})",
                    'Test AUC': f"{report['test_auc']:.4f}",
                    'Accuracy': f"{report['test_accuracy']:.2%}",
                    'Precision': f"{report['test_precision']:.2%}",
                    'Recall': f"{report['test_recall']:.2%}",
                    'F1 Score': f"{report['test_f1']:.2%}",
                })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, width='stretch', hide_index=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3>🔧 Best Model Configuration</h3>', unsafe_allow_html=True)

        best_params = artifacts.get('best_params', {})
        params_df = pd.DataFrame([
            {'Parameter': k.replace('_', ' ').title(), 'Value': str(v)}
            for k, v in best_params.items()
        ])
        st.dataframe(params_df, width='stretch', hide_index=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if 'feature_importances' in model_report:
            st.markdown('<h3>⭐ Feature Importance</h3>', unsafe_allow_html=True)
            feat_df = model_report['feature_importances']

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            top_n = feat_df.head(15)
            colors_imp = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_n)))
            bars = ax.barh(top_n['feature'][::-1], top_n['importance'][::-1], color=colors_imp[::-1], height=0.6)

            ax.set_xlabel('Importance', color='#a0a0a0', fontsize=12)
            ax.set_title(f'Top Feature Importances ({best_model_name})', color='white',
                         fontsize=14, fontweight='bold')
            ax.tick_params(colors='#e0e0e0', labelsize=11)
            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))

            st.pyplot(fig)

    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        <div style="color: #e0e0e0;">
            <p><strong>Training Pipeline:</strong></p>
            <ul>
                <li>Multiple models trained: Logistic Regression, Random Forest, Gradient Boosting, XGBoost</li>
                <li>5-fold cross-validation for robust evaluation</li>
                <li>Grid search hyperparameter tuning on the best model</li>
                <li>Stratified train/test split (80/20)</li>
            </ul>
            <p><strong>Features (19 total):</strong> Demographics (gender, SeniorCitizen), Account Info (tenure,
            contract, payment method, charges), Services (phone, internet, security, backup, etc.)</p>
            <p><strong>Data:</strong> {} customers with {}% churn rate</p>
        </div>
        """.format(len(load_data()), round(load_data()['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)),
        unsafe_allow_html=True)


