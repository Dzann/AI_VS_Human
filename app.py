import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

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
    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Fitur Pendukung")

with st.sidebar.expander("📖 Penjelasan Fitur", expanded=False):

    st.markdown("""
    **Prompt Complexity Score**
    
    Tingkat kerumitan isi teks.

    **Perplexity Score**
    
    Tingkat kealamian teks saat dibaca.

    **Burstiness Index**
    
    Variasi panjang kalimat dalam tulisan.

    **Syntactic Variability**
    
    Variasi struktur atau susunan kalimat.

    **Semantic Coherence**
    
    Keterkaitan antar kalimat dalam teks.

    **Lexical Diversity**
    
    Keragaman kosakata yang digunakan.

    **Readability Grade**
    
    Tingkat kesulitan teks untuk dibaca.

    **Generation Confidence**
    
    Tingkat keyakinan model terhadap pola teks.
    """)

prompt_complexity_score = st.sidebar.number_input(
    "Prompt Complexity Score",
    value=0.5
)

perplexity_score = st.sidebar.number_input(
    "Perplexity Score",
    value=50.0
)

burstiness_index = st.sidebar.number_input(
    "Burstiness Index",
    value=0.5
)

syntactic_variability = st.sidebar.number_input(
    "Syntactic Variability",
    value=0.5
)

semantic_coherence_score = st.sidebar.number_input(
    "Semantic Coherence",
    value=0.5
)

lexical_diversity_ratio = st.sidebar.number_input(
    "Lexical Diversity",
    value=0.5
)

readability_grade_level = st.sidebar.number_input(
    "Readability Grade",
    value=10.0
)

generation_confidence_score = st.sidebar.number_input(
    "Generation Confidence",
    value=0.5
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
    placeholder="Tempel atau ketik teks yang ingin dianalisis di sini..."
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

        st.success(f"✅ Hasil Prediksi: {label}")

        st.info(
            f"🎯 Tingkat Keyakinan Model: {confidence:.2f}%"
        )

        st.progress(float(confidence / 100))
