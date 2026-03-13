import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set page config
st.set_page_config(
    page_title="Roblox Game Success Predictor",
    page_icon="🎮",
    layout="wide"
)

# =====================================================
# 1. LOAD MODEL & ARTIFACTS
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

preprocessor_pipeline = final_model.named_steps['preprocessor']
feature_selector_pipeline = final_model.named_steps['feature_selection']

fitur_numerik = ['game_age', 'update_gap_days', 'visit_velocity', 'favorite_rate', 'engagement_rate', 'like_ratio']
fitur_kategorik = ['Genre', 'AgeRecommendation']

unique_genres = df['Genre'].dropna().unique().tolist()
unique_age_recommendations = df['AgeRecommendation'].dropna().unique().tolist()

# =====================================================
# STREAMLIT 
# =====================================================
st.title("🚀 Roblox Game Success Predictor")
st.markdown("Selamat datang di aplikasi prediksi kesuksesan game Roblox!")

tab1, tab2, tab3 = st.tabs(["1️⃣ 🎮 Predictor", "2️⃣ 📊 Model Performance", "3️⃣ 📈 Data Insights"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.header("🎮 Prediksi Kesuksesan Game")
    st.markdown("Masukkan fitur-fitur game di bawah untuk memprediksi apakah game tersebut akan sukses atau tidak.")

    with st.form("prediction_form"):
        st.subheader("Informasi Game")

        col1, col2 = st.columns(2)

        with col1:
            genre = st.selectbox("Genre", options=unique_genres)
            age_recommendation = st.selectbox("Rekomendasi Usia", options=unique_age_recommendations)
            game_age = st.number_input("Usia Game (hari)", min_value=0, value=300)
            update_gap_days = st.number_input("Gap Hari Update Terakhir (hari)", min_value=0, value=60)

        with col2:
            visits = st.number_input("Jumlah Kunjungan (Visits)", min_value=0.0, value=100000.0, step=1000.0)
            favorites = st.number_input("Jumlah Favorit (Favorites)", min_value=0.0, value=5000.0, step=100.0)
            likes = st.number_input("Jumlah Suka (Likes)", min_value=0.0, value=10000.0, step=100.0)
            dislikes = st.number_input("Jumlah Tidak Suka (Dislikes)", min_value=0.0, value=1000.0, step=100.0)

        submitted = st.form_submit_button("Prediksi Kesuksesan")

        if submitted:
            # Re-calculating engineered features based on raw inputs
            calculated_game_age = game_age
            calculated_update_gap_days = update_gap_days

            # Handle division by zero for rates
            calculated_visit_velocity = visits / (calculated_game_age + 1) if (calculated_game_age + 1) != 0 else 0
            calculated_favorite_rate = favorites / (visits + 1) if (visits + 1) != 0 else 0
            calculated_engagement_rate = (likes + dislikes) / (visits + 1) if (visits + 1) != 0 else 0
            calculated_like_ratio = likes / (likes + dislikes + 1) if (likes + dislikes + 1) != 0 else 0

            # Apply log1p transformation as done in the notebook for some features
            calculated_visit_velocity = np.log1p(calculated_visit_velocity)
            calculated_favorite_rate = np.log1p(calculated_favorite_rate)
            calculated_engagement_rate = np.log1p(calculated_engagement_rate)
            calculated_update_gap_days = np.log1p(calculated_update_gap_days)

            # Create a DataFrame for prediction
            input_data = pd.DataFrame({
                'game_age': [calculated_game_age],
                'update_gap_days': [calculated_update_gap_days],
                'visit_velocity': [calculated_visit_velocity],
                'favorite_rate': [calculated_favorite_rate],
                'engagement_rate': [calculated_engagement_rate],
                'like_ratio': [calculated_like_ratio],
                'Genre': [genre],
                'AgeRecommendation': [age_recommendation]
            })

            # Predict
            prediction = final_model.predict(input_data)
            prediction_proba = final_model.predict_proba(input_data)[:, 1]

            st.subheader("Hasil Prediksi")
            if prediction[0] == 1:
                st.success(f"Game ini **diprediksi Sukses**! (Probabilitas Sukses: {prediction_proba[0]:.2f})")
            else:
                st.error(f"Game ini **diprediksi Tidak Sukses**. (Probabilitas Sukses: {prediction_proba[0]:.2f})")

# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.header("📊 Kinerja Model")
    st.markdown("Metrik evaluasi dan visualisasi kinerja dari model terbaik.")

    st.subheader("Metrik Evaluasi")
    st.json(metrics)

    st.subheader("Confusion Matrix")
    st.markdown("**Interpretasi Confusion Matrix:** Tabel ini menunjukkan performa model dalam mengklasifikasikan game sukses dan tidak sukses. Nilai pada diagonal (kiri atas dan kanan bawah) adalah prediksi yang benar, sedangkan nilai di luar diagonal adalah kesalahan prediksi.")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4)) 
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Sukses', 'Sukses'],
                yticklabels=['Tidak Sukses', 'Sukses'], ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_ylabel('Aktual')
    ax_cm.set_xlabel('Prediksi')
    st.pyplot(fig_cm)

    st.subheader("Receiver Operating Characteristic (ROC) Curve")
    st.markdown("**Interpretasi ROC Curve & AUC:** Kurva ROC menggambarkan kemampuan model untuk membedakan antara kelas positif dan negatif. Nilai AUC (Area Under the Curve) sebesar {:.2f} menunjukkan bahwa model memiliki kemampuan diskriminasi yang sangat baik; semakin dekat ke 1, semakin baik model dalam membedakan antara game sukses dan tidak sukses.".format(metrics["roc_auc"]))
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4)) 
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# --- TAB 3: DATA INSIGHTS ---
with tab3:
    st.header("📈 Data Insights")
    st.markdown("Visualisasi dan statistik deskriptif dari dataset.")

    st.subheader("Statistik Deskriptif untuk Kolom Numerik")
    st.markdown("**Interpretasi Statistik Deskriptif:** Tabel ini menyajikan ringkasan statistik dasar (rata-rata, standar deviasi, min, maks, kuartil) untuk semua fitur numerik dalam dataset, memberikan gambaran umum tentang distribusi dan rentang nilai data.")
    st.dataframe(df.describe())

    st.subheader("Visualisasi Distribusi Genre")
    st.markdown("**Interpretasi Distribusi Genre:** Bar chart ini menunjukkan frekuensi atau jumlah game untuk setiap genre yang ada dalam dataset. Ini membantu memahami popularitas relatif dari berbagai kategori game Roblox.")
    fig_genre, ax_genre = plt.subplots(figsize=(7, 5)) 
    sns.countplot(data=df, y='Genre', order=df['Genre'].value_counts().index, palette='viridis', ax=ax_genre, hue='Genre', legend=False)
    ax_genre.set_title('Distribution of Game Genres')
    ax_genre.set_xlabel('Count')
    ax_genre.set_ylabel('Genre')
    st.pyplot(fig_genre)

    st.subheader("Visualisasi Matriks Korelasi Pearson")
    st.markdown("**Interpretasi Matriks Korelasi Pearson:** Heatmap ini menampilkan koefisien korelasi Pearson antara setiap pasangan fitur numerik. Nilai mendekati 1 atau -1 menunjukkan hubungan linear yang kuat (positif atau negatif), sedangkan nilai mendekati 0 menunjukkan tidak ada hubungan linear. Ini membantu mengidentifikasi fitur yang saling bergantung.")
    fig_corr, ax_corr = plt.subplots(figsize=(9, 7)) 
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Pearson Correlation Matrix of Numerical Features')
    ax_corr.tick_params(axis='x', rotation=45)
    ax_corr.tick_params(axis='y', rotation=0)
    st.pyplot(fig_corr)

    if df_imp is not None:
        st.subheader("Top 10 Feature Importance (dari Random Forest)")
        st.markdown("**Interpretasi Feature Importance:** Bar chart ini menampilkan 10 fitur yang paling berkontribusi dalam prediksi model Random Forest. Semakin tinggi 'Importance', semakin besar pengaruh fitur tersebut terhadap hasil prediksi kesuksesan game.")
        fig_imp, ax_imp = plt.subplots(figsize=(7, 5)) 
        sns.barplot(x='Importance', y='Fitur', data=df_imp.head(10), palette='viridis', ax=ax_imp, hue='Fitur', legend=False)
        ax_imp.set_title('Top 10 Feature Importance - Random Forest')
        st.pyplot(fig_imp)
    else:
        st.info("Feature importance tidak tersedia karena model terbaik bukan Random Forest.")