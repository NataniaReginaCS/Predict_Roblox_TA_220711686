import streamlit as st
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Roblox Game Success Predictor", page_icon="🎮", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('model/model.pkl')

@st.cache_data
def load_metrics():
    with open('model/metrics.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    return pd.read_csv('model/feature_importance.csv')

model = load_model()
metrics = load_metrics()
df_imp = load_feature_importance()

# ==========================================
# APP HEADER
# ==========================================
st.title("🎮 Roblox Game Success Prediction")
st.markdown("Dashboard ini merupakan *deployment* dari model *Machine Learning* untuk memprediksi probabilitas kesuksesan game di platform Roblox.")

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3 = st.tabs(["1️⃣ 🎮 Predictor", "2️⃣ 📊 Model Performance", "3️⃣ 📈 Data Insights"])

# ------------------------------------------
# TAB 1: PREDICTOR (Interaktif)
# ------------------------------------------
with tab1:
    st.header("Predict Game Success")
    
    col1, col2 = st.columns(2)
    with col1:
        game_age = st.number_input("Game Age (Days)", min_value=0, value=100, step=1)
        favorite_rate = st.number_input("Favorite Rate (Favorites/Visits)", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
        engagement_rate = st.number_input("Engagement Rate (Likes vs Dislikes)", min_value=-1.0, max_value=1.0, value=0.70, format="%.4f")
    
    with col2:
        genre = st.selectbox("Genre", ['Action', 'RPG', 'Horror', 'Adventure', 'Fighting', 'Comedy', 'All'])
        age_rec = st.selectbox("Age Recommendation", ['All Ages', '9+', '13+', '17+'])
        
        st.markdown("---")

        threshold = st.slider("Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    if st.button("🚀 Predict Success", use_container_width=True):
        input_data = pd.DataFrame({
            'game_age': [game_age],
            'favorite_rate': [favorite_rate],
            'engagement_rate': [engagement_rate],
            'Genre': [genre],
            'AgeRecommendation': [age_rec]
        })

        prob = model.predict_proba(input_data)[0][1]
        is_success = prob >= threshold

        st.markdown("### Prediction Result")
        if is_success:
            st.success("🔥 **PREDICTION: SUCCESS**")
        else:
            st.error("❄️ **PREDICTION: NOT SUCCESS**")
            
        st.metric(label="Success Probability", value=f"{prob:.1%}")
        st.progress(float(prob))
        

        st.caption("ℹ️ *Success is defined as being in the top 20% of active games.*")

# ------------------------------------------
# TAB 2: MODEL PERFORMANCE 
# ------------------------------------------
with tab2:
    st.header("Model Evaluation")
    st.write("Performa model dievaluasi pada data uji (offline) menggunakan *stratified split*.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")

    col4.metric("🏆 F1-Score (Primary)", f"{metrics['f1_score']:.2%}") 
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    
    st.markdown("---")
    

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image('assets/confusion_matrix.png', caption="Confusion Matrix pada Data Uji", use_column_width=True)
    with col_img2:
        st.image('assets/roc_curve.png', caption="Receiver Operating Characteristic", use_column_width=True)

# ------------------------------------------
# TAB 3: DATA INSIGHTS
# ------------------------------------------
with tab3:
    st.header("Feature Importance")
    st.write("Faktor mana yang paling menentukan kesuksesan sebuah game?")
    
    top_10_imp = df_imp.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_10_imp, palette='viridis', ax=ax)
    plt.title("Top 10 Most Important Features", pad=10)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    
    st.pyplot(fig)
    
    with st.expander("Lihat Data Tabel"):
        st.dataframe(df_imp, use_container_width=True)