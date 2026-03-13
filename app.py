import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(86,171,47,0.3);
    }
    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255,107,107,0.3);
    }
    .stats-metric {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="🎮 Roblox Game Success Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL 
# =====================================================
@st.cache_resource
def load_model_and_artifacts():
    final_model = joblib.load('final_model.pkl')
    processed_df = joblib.load('processed_df.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    y_pred = joblib.load('y_pred.pkl')
    y_prob = joblib.load('y_prob.pkl')
    metrics = joblib.load('metrics.pkl')
    roc_data = joblib.load('roc_data.pkl')

    df_imp = None
    try:
        df_imp = joblib.load('feature_importance_df.pkl')
    except FileNotFoundError:
        pass

    return final_model, processed_df, X_test, y_test, y_pred, y_prob, metrics, roc_data, df_imp

final_model, df, X_test, y_test, y_pred, y_prob, metrics, roc_data, df_imp = load_model_and_artifacts()

# Fitur 
fitur_numerik = ['game_age', 'update_gap_days', 'visit_velocity', 'favorite_rate', 'engagement_rate', 'like_ratio']
fitur_kategorik = ['Genre', 'AgeRecommendation']
unique_genres = df['Genre'].dropna().unique().tolist()
unique_age_recommendations = df['AgeRecommendation'].dropna().unique().tolist()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🎯 Kontrol")
    st.info("**Model Terbaik:** Random Forest")
    st.success(f"**F1-Score Test:** {metrics['f1_score']:.4f}")
    st.caption("Aplikasi prediksi kesuksesan game Roblox berbasis ML")

# =====================================================
# MAIN PAGE
# =====================================================
st.markdown('<h1 class="main-header">🚀 Roblox Game Success Predictor</h1>', unsafe_allow_html=True)
st.markdown("### *Prediksi tingkat kesuksesan game Roblox menggunakan Random Forest*")

col1, col2 = st.columns([3, 1])
with col2:
    st.markdown("### 📈 **Quick Stats**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}", delta=f"+{metrics['f1_score']:.1%}")
    with col_b:
        st.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")

tab1, tab2, tab3 = st.tabs(["🎮 **Prediksi Game**", "📊 **Model Analytics**", "📈 **Data Explorer**"])

# =====================================================
# TAB 1: PREDICTOR
# =====================================================
with tab1:
    st.markdown("---")
    
    # Input form dengan gradient cards
    with st.form("prediction_form", clear_on_submit=True):
        st.markdown("#### 🔧 **Input Data Game**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 Informasi Dasar**")
            genre = st.selectbox("🎨 Genre", options=unique_genres, 
                                help="Pilih kategori genre game")
            age_rec = st.selectbox("👶 Rekomendasi Usia", options=unique_age_recommendations)
            
        with col2:
            st.markdown("**📊 Metrik Engagement**")
            game_age = st.number_input("📅 Usia Game (hari)", min_value=0, value=300)
            visits = st.number_input("👥 Total Visits", min_value=0.0, value=100000.0, step=1000.0)
            favorites = st.number_input("❤️ Favorites", min_value=0.0, value=5000.0, step=100.0)
            
        col3, col4 = st.columns(2)
        with col3:
            likes = st.number_input("👍 Likes", min_value=0.0, value=10000.0, step=100.0)
            dislikes = st.number_input("👎 Dislikes", min_value=0.0, value=1000.0, step=100.0)
        with col4:
            update_gap = st.number_input("🔄 Gap Update (hari)", min_value=0, value=60)
        
        submitted = st.form_submit_button("🎯 **Prediksi Sekarang**", 
                                        use_container_width=True,
                                        help="Klik untuk memproses prediksi")

    if submitted:
        # Calculate engineered features
        visit_velocity = np.log1p(visits / (game_age + 1))
        favorite_rate = np.log1p(favorites / (visits + 1))
        engagement_rate = np.log1p((likes + dislikes) / (visits + 1))
        like_ratio = likes / (likes + dislikes + 1)
        update_gap_days = np.log1p(update_gap)

        input_data = pd.DataFrame({
            'game_age': [game_age],
            'update_gap_days': [update_gap_days],
            'visit_velocity': [visit_velocity],
            'favorite_rate': [favorite_rate],
            'engagement_rate': [engagement_rate],
            'like_ratio': [like_ratio],
            'Genre': [genre],
            'AgeRecommendation': [age_rec]
        })

        # Predict
        pred = final_model.predict(input_data)[0]
        prob_success = final_model.predict_proba(input_data)[:, 1][0]

        # Beautiful result display
        st.markdown("---")
        st.markdown("#### 🎉 **Hasil Prediksi**")
        
        if pred == 1:
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ **GAME SUKSES!**</h2>
                <h3>🎊 Probabilitas: <span class="stats-metric">{prob_success:.1%}</span></h3>
                <p>Game ini memiliki potensi tinggi untuk menjadi hits di Roblox!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h2>❌ **Tidak Sukses**</h2>
                <h3>📉 Probabilitas Sukses: <span class="stats-metric">{prob_success:.1%}</span></h3>
                <p>Perlu optimasi strategi untuk meningkatkan performa.</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# TAB 2: MODEL PERFORMANCE
# =====================================================
with tab2:
    st.markdown("---")
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>F1-Score</h3>
            <h1>{:.3f}</h1>
        </div>
        """.format(metrics['f1_score']), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h1>{:.1%}</h1>
        </div>
        """.format(metrics['accuracy']), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>AUC-ROC</h3>
            <h1>{:.3f}</h1>
        </div>
        """.format(metrics['roc_auc']), unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Precision</h3>
            <h1>{:.3f}</h1>
        </div>
        """.format(metrics['precision']), unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['❌ Tidak Sukses', '✅ Sukses'],
                    yticklabels=['❌ Aktual', '✅ Aktual'], ax=ax,
                    cbar_kws={'label': 'Jumlah'})
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)
    
    with col2:
        st.subheader("📈 ROC Curve")
        fig_roc, ax = plt.subplots(figsize=(6, 5))
        fpr, tpr = roc_data['fpr'], roc_data['tpr']
        ax.plot(fpr, tpr, color='#1f77b4', lw=3, 
                label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.7)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        st.pyplot(fig_roc)

# =====================================================
# TAB 3: DATA INSIGHTS
# =====================================================
with tab3:
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Distribusi Genre")
        fig_genre, ax = plt.subplots(figsize=(8, 5))
        genre_counts = df['Genre'].value_counts().head(10)
        colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
        bars = ax.barh(genre_counts.index, genre_counts.values, color=colors)
        ax.set_xlabel('Jumlah Game'); ax.set_title('Top 10 Game Genres', fontweight='bold')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_genre)
    
    with col2:
        if df_imp is not None:
            st.subheader("🏆 Top Features")
            fig_imp, ax = plt.subplots(figsize=(8, 5))
            top_features = df_imp.head(8)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                            color=colors, alpha=0.8)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Fitur'])
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 8 Feature Importance', fontweight='bold')
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, i, f'{width:.3f}', va='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_imp)
        else:
            st.info("📊 Feature importance tidak tersedia")

    st.subheader("📋 Dataset Overview")
    st.dataframe(df.describe()[['mean', 'std', 'min', 'max']].round(3), 
                use_container_width=True)