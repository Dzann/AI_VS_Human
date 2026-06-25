import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("random_forest_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ==========================
# CLEAN TEXT
# ==========================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI vs Human Detector",
    page_icon="🤖",
    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Fitur Pendukung")

prompt_complexity_score = st.sidebar.number_input(
    "Prompt Complexity Score",
    value=0.5,
    help="Mengukur tingkat kerumitan isi teks. Nilai yang lebih tinggi menunjukkan teks lebih kompleks dan detail."
)

perplexity_score = st.sidebar.number_input(
    "Perplexity Score",
    value=50.0,
    help="Mengukur seberapa alami suatu teks ketika dibaca. Digunakan untuk melihat pola penulisan yang lebih terstruktur atau spontan."
)

burstiness_index = st.sidebar.number_input(
    "Burstiness Index",
    value=0.5,
    help="Mengukur variasi panjang kalimat dalam tulisan. Nilai tinggi menunjukkan adanya campuran kalimat pendek dan panjang."
)

syntactic_variability = st.sidebar.number_input(
    "Syntactic Variability",
    value=0.5,
    help="Mengukur variasi struktur atau susunan kalimat yang digunakan dalam teks."
)

semantic_coherence_score = st.sidebar.number_input(
    "Semantic Coherence",
    value=0.5,
    help="Mengukur keterkaitan antar kalimat dalam sebuah tulisan. Nilai tinggi menunjukkan topik lebih konsisten."
)

lexical_diversity_ratio = st.sidebar.number_input(
    "Lexical Diversity",
    value=0.5,
    help="Mengukur keragaman kosakata yang digunakan. Semakin tinggi berarti semakin banyak variasi kata."
)

readability_grade_level = st.sidebar.number_input(
    "Readability Grade",
    value=10.0,
    help="Mengukur tingkat kesulitan teks untuk dibaca dan dipahami."
)

generation_confidence_score = st.sidebar.number_input(
    "Generation Confidence",
    value=0.5,
    help="Menunjukkan seberapa kuat pola teks yang dikenali oleh model untuk membantu proses klasifikasi."
)

# ==========================
# MAIN PAGE
# ==========================
st.title("🤖 AI vs Human Text Detection")

st.write(
    "Deteksi apakah sebuah teks ditulis oleh AI atau manusia menggunakan algoritma Random Forest."
)

text = st.text_area(
    "Masukkan Teks",
    height=250,
    placeholder="Tempel atau ketik teks yang ingin dianalisis dalam bahasa inggris di sini..."
)

# ==========================
# PREDIKSI
# ==========================
if st.button("🔍 Prediksi", use_container_width=True):

    if text.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:

        text_clean = clean_text(text)

        text_vector = tfidf.transform([text_clean])

        numeric_data = np.array([
            [
                prompt_complexity_score,
                perplexity_score,
                burstiness_index,
                syntactic_variability,
                semantic_coherence_score,
                lexical_diversity_ratio,
                readability_grade_level,
                generation_confidence_score
            ]
        ])

        numeric_sparse = csr_matrix(numeric_data)

        final_input = hstack([
            text_vector,
            numeric_sparse
        ])

        pred = model.predict(final_input)

        prob = model.predict_proba(final_input)

        label = le.inverse_transform(pred)[0]

        confidence = np.max(prob) * 100

        if label.lower() == "ai":
            st.error(f"🤖 Hasil Prediksi: {label}")
        else:
            st.success(f"👤 Hasil Prediksi: {label}")
    # ==================================
# ANALISIS HASIL KLASIFIKASI
# ==================================

st.subheader("📈 Analisis Hasil Klasifikasi")

prob_df = pd.DataFrame({
    "Kelas": le.classes_,
    "Probabilitas": prob[0] * 100
})

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(
    prob_df["Kelas"],
    prob_df["Probabilitas"]
)

ax.set_ylabel("Persentase (%)")
ax.set_title("Distribusi Probabilitas Prediksi")

st.pyplot(fig)

st.dataframe(
    prob_df,
    use_container_width=True
)
        st.info(
            f"🎯 Tingkat Keyakinan Model: {confidence:.2f}%"
        )

        # ==================================
# FEATURE IMPORTANCE
# ==================================

st.subheader("📊 Analisis Feature Importance")

feature_names = [
    "Prompt Complexity",
    "Perplexity",
    "Burstiness",
    "Syntactic Variability",
    "Semantic Coherence",
    "Lexical Diversity",
    "Readability Grade",
    "Generation Confidence"
]

importances = model.feature_importances_[-8:]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")

st.pyplot(fig)

st.dataframe(
    importance_df.sort_values(
        by="Importance",
        ascending=False
    ),
    use_container_width=True
)

        st.progress(confidence / 100)

        st.subheader("Ringkasan Hasil")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Prediksi",
                value=label
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )
