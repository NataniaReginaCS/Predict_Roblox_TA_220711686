import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="🎮 Roblox Game Success Predictor",
    page_icon="🎮",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main-header {
    font-size: clamp(2rem, 5vw, 3rem) !important;
    font-weight: 800 !important;
    color: #1f77b4 !important;
    text-align: center;
    margin: 1rem 0 2rem 0 !important;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102,126,234,0.3);
}
.success-box {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(86,171,47,0.3);
}
.error-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(255,107,107,0.3);
}
.stButton > button {
    background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 20px rgba(31,119,180,0.3);
}
@media (max-width: 768px) {
    .metric-card { padding: 1rem !important; }
    .success-box, .error-box { padding: 1.5rem !important; }
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
        metrics = joblib.load("metrics.pkl")
        roc_data = joblib.load("roc_data.pkl")
        
        # feature importance
        df_imp = pd.DataFrame()
        try:
            df_imp = joblib.load("feature_importance_df.pkl")
        except:
            pass
            
        return final_model, df, X_test, y_test, y_pred, metrics, roc_data, df_imp
    except Exception as e:
        st.error(f"🚨 Model files missing. Upload: final_model.pkl, processed_df.pkl, etc.")
        st.stop()

# Load 
final_model, df, X_test, y_test, y_pred, metrics, roc_data, df_imp = load_model_and_artifacts()
baseline_prob = float(np.mean(y_test))

# Data prep
unique_genres = sorted(df.get("Genre", pd.Series()).dropna().unique().tolist())
unique_ages = sorted(df.get("AgeRecommendation", pd.Series()).dropna().unique().tolist())

# =====================================================
# HEADER
# =====================================================
st.markdown('<h1 class="main-header">🚀 Roblox Game Success Predictor</h1>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; color:#666; font-size:1.2rem; margin-bottom:2rem;'>
    AI-powered prediction using <strong>Random Forest</strong> | F1: {metrics['f1_score']:.3f}
</div>
""", unsafe_allow_html=True)

st.markdown("### 📊 Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯 F1 Score</h3>
        <h2>{metrics['f1_score']:.3f}</h2>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>✅ Accuracy</h3>
        <h2>{metrics['accuracy']:.1%}</h2>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 AUC</h3>
        <h2>{metrics['roc_auc']:.3f}</h2>
    </div>""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎮 Predict Game", "📊 Analytics", "📈 Data"])

# =====================================================
# TAB 1: PREDICTOR
# =====================================================
with tab1:
    st.markdown("---")
    
    with st.form("predict_form", clear_on_submit=True):
        # Responsive form
        col1, col2 = st.columns(2)
        with col1:
            genre = st.selectbox("🎨 Genre", unique_genres)
            age_rec = st.selectbox("👶 Age Rating", unique_ages)
            game_age = st.number_input("📅 Game Age (days)", 0, 3650, 300)
        with col2:
            visits = st.number_input("👥 Total Visits", 0.0, 10000000.0, 50000.0)
            favorites = st.number_input("❤️ Favorites", 0, 1000000, 2500)
            update_gap = st.number_input("🔄 Update Gap (days)", 0, 365, 30)
        
        col3, col4 = st.columns(2)
        with col3:
            likes = st.number_input("👍 Likes", 0, 1000000, 5000)
        with col4:
            dislikes = st.number_input("👎 Dislikes", 0, 100000, 500)
        
        submitted = st.form_submit_button("🚀 ANALYZE GAME", use_container_width=True)
    
    if submitted:
        with st.spinner("Analyzing..."):
            # feature engineering
            visit_velocity = np.log1p(visits / max(game_age, 1))
            favorite_rate = np.log1p(favorites / max(visits, 1))
            engagement_rate = np.log1p((likes + dislikes) / max(visits, 1))
            like_ratio = likes / max(likes + dislikes, 1)
            update_gap_days = np.log1p(update_gap)

            input_df = pd.DataFrame({
                "game_age": [game_age],
                "update_gap_days": [update_gap_days],
                "visit_velocity": [visit_velocity],
                "favorite_rate": [favorite_rate],
                "engagement_rate": [engagement_rate],
                "like_ratio": [like_ratio],
                "Genre": [genre],
                "AgeRecommendation": [age_rec]
            })

            # Predict
            prediction = final_model.predict(input_df)[0]
            probability = final_model.predict_proba(input_df)[0, 1]

        # Progress bar
        st.progress(min(probability, 1.0))
        
        # Result
        if prediction == 1:
            st.markdown(f"""
            <div class="success-box">
                <h2>🎉 SUCCESS PREDICTED!</h2>
                <h3>Success Probability: <strong>{probability:.1%}</strong></h3>
                <p>🚀 This game has **viral potential**!</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h2>⚠️ Optimization Needed</h2>
                <h3>Success Probability: <strong>{probability:.1%}</strong></h3>
                <p>💡 Improve engagement & marketing</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 🧠 **Why This Prediction?**")
        
        engagement_rate = (likes + dislikes) / max(visits, 1)
        sentiment = likes / max(likes + dislikes, 1)
        retention = favorites / max(visits, 1)
        velocity = visits / max(game_age, 1)
        
        # Display insights
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Engagement Rate", f"{engagement_rate:.1%}")
            st.metric("❤️ Retention Rate", f"{retention:.2%}")
        with col2:
            st.metric("👍 Sentiment", f"{sentiment:.1%}")
            st.metric("📈 Daily Velocity", f"{velocity:.0f}/day")
        
        st.markdown("---")
        st.markdown("""
        **🟢 Excellent** (>80%) | **🟡 Good** (50-80%) | **🔴 Needs Work** (<50%)
        """)

# =====================================================
# TAB 2: ANALYTICS
# =====================================================
with tab2:
    # Target distribution
    st.subheader("📊 Dataset Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Not Success', 'Success']
    sizes = [len(y_test[y_test==0]), len(y_test[y_test==1])]
    colors = ['#ff9999', '#66b3ff']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                        autopct='%1.1f%%', startangle=90)
    ax.set_title('Success Rate in Dataset', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    
    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Success', 'Success'],
                    yticklabels=['Actual Not', 'Actual Yes'], ax=ax)
        ax.set_title('Prediction Results')
        st.pyplot(fig)
    
    # Simple ROC
    with col2:
        st.subheader("📈 Model Strength")
        st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")

# =====================================================
# TAB 3: DATA EXPLORER
# =====================================================
with tab3:
    st.subheader("📈 Dataset Overview")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Games", f"{len(df):,}")
    with col2: st.metric("Features", df.shape[1])
    with col3: st.metric("Genres", df['Genre'].nunique())
    
    # Genre distribution
    st.subheader("🎨 Top Genres")
    genre_counts = df['Genre'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(genre_counts.index, genre_counts.values, color='skyblue', alpha=0.8)
    ax.set_xlabel('Number of Games')
    ax.set_title('Most Common Game Genres')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 50, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', va='center')
    st.pyplot(fig)
    
    # Dataset preview
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(), use_container_width=True)
    
    # Feature importance 
    if not df_imp.empty:
        st.subheader("🏆 Top Features")
        top_features = df_imp.head(10)
        st.bar_chart(top_features.set_index('Fitur')['Importance'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; padding:2rem;'>
    🎓 Machine Learning Project | Random Forest Model
</div>
""", unsafe_allow_html=True)