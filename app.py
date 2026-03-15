import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🎮 Roblox Game Success Predictor",
    page_icon="🎮",
    layout="wide"
)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# PREMIUM CUSTOM CSS (Mobile-First)
# =====================================================
st.markdown("""
<style>
/* Header */
.main-header {
    font-size: clamp(2rem, 5vw, 3rem) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin: 1rem 0 2rem 0 !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    transition: transform 0.3s ease;
    border: 1px solid rgba(255,255,255,0.2);
}
.metric-card:hover {
    transform: translateY(-5px);
}

/* Result Boxes */
.success-box {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(86,171,47,0.3);
    border: 3px solid rgba(255,255,255,0.3);
}
.error-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(255,107,107,0.3);
    border: 3px solid rgba(255,255,255,0.3);
}

/* Stats */
.stats-metric {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #1f77b4 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(31,119,180,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(31,119,180,0.4);
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .metric-card { padding: 1rem 0.75rem !important; margin: 0.5rem 0; }
    .success-box, .error-box { padding: 1.5rem 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL 
# =====================================================
@st.cache_resource
def load_model_and_artifacts():
    try:
        final_model = joblib.load("final_model.pkl")
        df = joblib.load("processed_df.pkl")
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        y_pred = joblib.load("y_pred.pkl")
        y_prob = joblib.load("y_prob.pkl")
        metrics = joblib.load("metrics.pkl")
        roc_data = joblib.load("roc_data.pkl")

        df_imp = None
        try:
            df_imp = joblib.load("feature_importance_df.pkl")
        except:
            pass

        return final_model, df, X_test, y_test, y_pred, y_prob, metrics, roc_data, df_imp
    except Exception as e:
        st.error(f"🚨 **Model loading failed:** {str(e)}")
        st.stop()

final_model, df, X_test, y_test, y_pred, y_prob, metrics, roc_data, df_imp = load_model_and_artifacts()

# Data prep
unique_genres = sorted(df.get("Genre", pd.Series()).dropna().unique().tolist())
unique_ages = sorted(df.get("AgeRecommendation", pd.Series()).dropna().unique().tolist())

# =====================================================
# HEADER
# =====================================================
st.markdown('<h1 class="main-header">🚀 Roblox Game Success Predictor</h1>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; color:#666; font-size:1.2rem; margin-bottom:2rem;'>
    Prediksi kesuksesan game Roblox menggunakan <strong>Random Forest</strong><br>
    <span style='color:#1f77b4; font-size:1.5rem;'>F1-Score: {metrics['f1_score']:.3f} | Accuracy: {metrics['accuracy']:.1%}</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PERFORMANCE METRICS
# =====================================================
st.markdown("### 📊 **Model Performance Overview**")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 F1-Score</h3>
        <h2 style='font-size:2.5rem; font-weight:900;'>{metrics['f1_score']:.3f}</h2>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>✅ Accuracy</h3>
        <h2 style='font-size:2.5rem; font-weight:900;'>{metrics['accuracy']:.1%}</h2>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 ROC-AUC</h3>
        <h2 style='font-size:2.5rem; font-weight:900;'>{metrics['roc_auc']:.3f}</h2>
    </div>""", unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🎮 **Game Predictor**", "📊 **Model Analytics**", "📈 **Data Explorer**"])

# =====================================================
# TAB 1: PREDICTOR 
# =====================================================
with tab1:
    st.markdown("---")
    st.markdown("#### 🔮 **Input Game Data**")
    
    with st.form("predict_form", clear_on_submit=True):
        col1, col2 = st.columns([1,1])
        with col1:
            genre = st.selectbox("🎨 **Genre**", options=unique_genres)
            age_rec = st.selectbox("👶 **Age Rating**", options=unique_ages)
        with col2:
            game_age = st.number_input("📅 **Game Age** (days)", min_value=0, value=300)
            update_gap = st.number_input("🔄 **Update Gap** (days)", min_value=0, value=30)
        
        col3, col4 = st.columns([1,1])
        with col3:
            visits = st.number_input("👥 **Total Visits**", min_value=0.0, value=50000.0)
            favorites = st.number_input("❤️ **Favorites**", min_value=0, value=2500)
        with col4:
            likes = st.number_input("👍 **Likes**", min_value=0, value=5000)
            dislikes = st.number_input("👎 **Dislikes**", min_value=0, value=500)
        
        submitted = st.form_submit_button("🚀 **ANALYZE GAME**", use_container_width=True)
    
    if submitted:
        with st.spinner("🔬 Analyzing game potential..."):
            # Feature engineering 
            visit_velocity = np.log1p(visits / max(game_age, 1))
            favorite_rate = np.log1p(favorites / max(visits, 1))
            engagement_rate = np.log1p((likes + dislikes) / max(visits, 1))
            like_ratio = likes / max(likes + dislikes, 1)
            update_gap_days = np.log1p(update_gap)

            input_df = pd.DataFrame({
                "game_age": [game_age], "update_gap_days": [update_gap_days],
                "visit_velocity": [visit_velocity], "favorite_rate": [favorite_rate],
                "engagement_rate": [engagement_rate], "like_ratio": [like_ratio],
                "Genre": [genre], "AgeRecommendation": [age_rec]
            })

            pred = final_model.predict(input_df)[0]
            prob = final_model.predict_proba(input_df)[:, 1][0]

        # Progress bar
        st.progress(min(prob, 1.0))
        st.success(f"**Success Probability: {prob:.1%}**")
        
        # Display result
        if pred == 1:
            st.markdown(f"""
            <div class="success-box">
                <h2>🎉 **HIGH SUCCESS POTENTIAL!**</h2>
                <h3 style='color:#2d5a2a; font-size:2rem;'>
                    <span class="stats-metric">{prob:.1%}</span>
                </h3>
                <p>🚀 This game has **viral potential** on Roblox!</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h2>⚠️ **Needs Optimization**</h2>
                <h3 style='color:#8b1a1a; font-size:2rem;'>
                    <span class="stats-metric">{prob:.1%}</span>
                </h3>
                <p>💡 Improve marketing & engagement strategies</p>
            </div>""", unsafe_allow_html=True)

# =====================================================
# TAB 2: ANALYTICS 
# =====================================================
with tab2:
    st.markdown("---")
    
    # Target Distribution
    st.subheader("📊 **Target Distribution**")
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = pd.Series(y_test).value_counts().sort_index()
    colors = ['#ff9999', '#66b3ff']
    bars = ax.bar(target_counts.index, target_counts.values, 
                    color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Success', 'Success'], fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Game Success Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Confusion Matrix + ROC 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 **Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(7, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Success', 'Success'],
                    yticklabels=['Actual Not', 'Actual Yes'],
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', fontweight='bold', pad=20)
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 **ROC Curve**")
        fig, ax = plt.subplots(figsize=(7, 5))
        fpr, tpr = roc_data["fpr"], roc_data["tpr"]
        ax.plot(fpr, tpr, color='#1f77b4', lw=3, 
                label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        ax.plot([0,1], [0,1], color='gray', lw=2, linestyle='--', alpha=0.6)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_title('ROC Curve Analysis', fontweight='bold', pad=20)
        st.pyplot(fig)

# =====================================================
# TAB 3: DATA EXPLORER
# =====================================================
with tab3:
    st.markdown("---")
    
    # Dataset Cards
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 Total Samples", f"{len(df):,}")
    with col2: st.metric("🔧 Features", df.shape[1])
    with col3: st.metric("🎨 Unique Genres", df["Genre"].nunique())
    
    st.markdown("---")
    
    # Top Genres 
    st.subheader("🎨 **Top 10 Genres**")
    genre_df = (df["Genre"].value_counts()
                .head(10)
                .reset_index()
                .rename(columns={'index': 'Genre', 'Genre': 'Count'}))
    st.dataframe(genre_df.style.background_gradient(cmap='viridis'), 
                use_container_width=True, height=300)
    
    # Dataset Statistics 
    st.subheader("📈 **Dataset Statistics**")
    try:
        desc = df.describe(include='all').T.round(2)
        st.dataframe(desc, use_container_width=True, height=400)
    except Exception as e:
        st.warning("📊 Dataset summary temporarily unavailable")
    
    # Feature Importance 
    if df_imp is not None and len(df_imp) > 0:
        st.subheader("🏆 **Top 10 Feature Importance**")
        top_imp = df_imp.nlargest(10, 'Importance')[['Fitur', 'Importance']]
        top_imp['Importance'] = top_imp['Importance'].round(4)
        st.dataframe(top_imp.style.background_gradient(cmap='plasma'), 
                    use_container_width=True)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        top10 = top_imp.sort_values('Importance')
        colors = plt.cm.plasma(np.linspace(0, 1, len(top10)))
        bars = ax.barh(range(len(top10)), top10['Importance'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels([str(f)[:25] + '...' if len(str(f)) > 25 else str(f) 
                            for f in top10['Fitur']], fontsize=10)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title('Feature Importance Ranking (Random Forest)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.0005, i, f'{width:.4f}', 
                    va='center', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; padding:2rem;'>
    <h3>🎓 Dibuat untuk Analisis Data Roblox</h3>
    <p>Model Random Forest | F1-Score {:.3f} | Powered by Streamlit</p>
</div>
""".format(metrics['f1_score']), unsafe_allow_html=True)