import streamlit as st
import joblib
import pandas as pd

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    .info-card h3 {
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Result cards */
    .result-churn {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(255, 75, 43, 0.4);
        animation: pulse 2s infinite;
    }
    
    .result-stay {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(56, 239, 125, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .result-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-desc {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Input styling */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
    }
    
    /* Feature importance styling */
    .feature-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .feature-name {
        color: rgba(255,255,255,0.8);
    }
    
    .feature-value {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model_churn.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="main-header">
    <h1>📊 Telco Customer Churn Predictor</h1>
    <p>Prediksi kemungkinan pelanggan akan berhenti berlangganan menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR - INFO
# ===============================
with st.sidebar:
    st.markdown("## 📋 Tentang Aplikasi")
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Tujuan</h3>
        <p>Memprediksi apakah pelanggan berpotensi churn (berhenti berlangganan) atau tetap berlangganan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>🤖 Model</h3>
        <p>Random Forest Classifier dengan akurasi pengujian ~77%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>📊 Dataset</h3>
        <p>IBM Telco Customer Churn Dataset dengan 7,043 pelanggan</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    **Faktor Risiko Churn Tinggi:**
    - Kontrak bulanan (Month-to-month)
    - Layanan Fiber optic
    - Tenure rendah (< 12 bulan)
    - Biaya bulanan tinggi
    """)

# ===============================
# MAIN CONTENT
# ===============================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📝 Input Data Pelanggan")
    
    # Create 3 columns for numeric inputs
    num_col1, num_col2, num_col3 = st.columns(3)
    
    with num_col1:
        tenure = st.number_input(
            "📅 Tenure (bulan)",
            min_value=0,
            max_value=100,
            value=12,
            help="Berapa lama pelanggan telah berlangganan (dalam bulan)"
        )
    
    with num_col2:
        monthly_charges = st.number_input(
            "💵 Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            format="%.2f",
            help="Biaya bulanan pelanggan"
        )
    
    with num_col3:
        total_charges = st.number_input(
            "💰 Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            format="%.2f",
            help="Total biaya yang telah dibayarkan"
        )
    
    st.markdown("---")
    
    # Create 3 columns for categorical inputs
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        paperless = st.selectbox(
            "📧 Paperless Billing",
            ["Yes", "No"],
            help="Apakah menggunakan tagihan tanpa kertas?"
        )
    
    with cat_col2:
        contract = st.selectbox(
            "📜 Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="Jenis kontrak berlangganan"
        )
    
    with cat_col3:
        internet = st.selectbox(
            "🌐 Internet Service",
            ["DSL", "Fiber optic", "No"],
            help="Jenis layanan internet"
        )
    
    st.markdown("---")
    
    # Predict button
    predict_btn = st.button("🔮 Prediksi Churn", use_container_width=True)

with col2:
    st.markdown("### 📊 Ringkasan Data")
    
    # Display metrics
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Tenure", value=f"{tenure} bulan")
        st.metric(label="Contract", value=contract)
    with metric_col2:
        st.metric(label="Monthly", value=f"${monthly_charges:.2f}")
        st.metric(label="Internet", value=internet)
    
    # Risk indicators
    st.markdown("### ⚠️ Indikator Risiko")
    
    risk_score = 0
    risk_factors = []
    
    if contract == "Month-to-month":
        risk_score += 30
        risk_factors.append("Kontrak bulanan")
    if internet == "Fiber optic":
        risk_score += 20
        risk_factors.append("Fiber optic service")
    if tenure < 12:
        risk_score += 25
        risk_factors.append("Tenure rendah")
    if monthly_charges > 70:
        risk_score += 15
        risk_factors.append("Biaya tinggi")
    if paperless == "Yes":
        risk_score += 10
        risk_factors.append("Paperless billing")
    
    # Display risk meter
    risk_color = "#38ef7d" if risk_score < 40 else "#ffc107" if risk_score < 70 else "#ff4b2b"
    st.progress(min(risk_score, 100))
    st.markdown(f"**Skor Risiko: {risk_score}%**")
    
    if risk_factors:
        st.warning("Faktor risiko: " + ", ".join(risk_factors))
    else:
        st.success("Tidak ada faktor risiko signifikan")

# ===============================
# PREDICTION
# ===============================
st.markdown("---")

if predict_btn:
    # Prepare input
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'PaperlessBilling': [paperless],
        'Contract': [contract],
        'InternetService': [internet]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Try to get probability
    try:
        proba = model.predict_proba(input_data)[0]
        churn_prob = proba[1] * 100
        stay_prob = proba[0] * 100
    except:
        churn_prob = 100 if prediction == 1 else 0
        stay_prob = 100 if prediction == 0 else 0
    
    # Display result
    st.markdown("### 🎯 Hasil Prediksi")
    
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    
    with result_col2:
        if prediction == 1:
            st.markdown("""
            <div class="result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-title">POTENSI CHURN</div>
                <div class="result-desc">Pelanggan ini berpotensi berhenti berlangganan</div>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"🔴 Probabilitas Churn: **{churn_prob:.1f}%**")
            
            st.markdown("#### 💡 Rekomendasi Tindakan:")
            st.markdown("""
            - Tawarkan diskon atau promosi khusus
            - Hubungi pelanggan untuk feedback
            - Berikan upgrade layanan gratis
            - Pertimbangkan kontrak jangka panjang dengan insentif
            """)
        else:
            st.markdown("""
            <div class="result-stay">
                <div class="result-icon">✅</div>
                <div class="result-title">TETAP BERLANGGANAN</div>
                <div class="result-desc">Pelanggan ini diprediksi akan tetap berlangganan</div>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"🟢 Probabilitas Tetap: **{stay_prob:.1f}%**")
            
            st.markdown("#### 💡 Rekomendasi:")
            st.markdown("""
            - Pertahankan kualitas layanan
            - Tawarkan program loyalitas
            - Upsell layanan tambahan
            """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); padding: 2rem;">
    <p>Built with Streamlit | Telco Customer Churn Prediction</p>
</div>
""", unsafe_allow_html=True)
