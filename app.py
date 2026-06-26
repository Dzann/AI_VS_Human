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
    help="Mengukur tingkat kompleksitas prompt atau teks. Nilai yang lebih tinggi menunjukkan struktur kalimat dan instruksi yang lebih kompleks."
)

perplexity_score = st.sidebar.number_input(
    "Perplexity Score",
    value=50.0,
    help="Mengukur seberapa sulit model bahasa memprediksi kata berikutnya. Nilai rendah biasanya menunjukkan pola yang lebih terprediksi."
)

burstiness_index = st.sidebar.number_input(
    "Burstiness Index",
    value=0.5,
    help="Mengukur variasi panjang dan pola kalimat dalam teks. Teks manusia umumnya memiliki burstiness yang lebih tinggi dibandingkan teks AI."
)

syntactic_variability = st.sidebar.number_input(
    "Syntactic Variability",
    value=0.5,
    help="Mengukur keragaman struktur tata bahasa yang digunakan dalam teks. Semakin tinggi nilainya, semakin beragam struktur kalimatnya."
)

semantic_coherence_score = st.sidebar.number_input(
    "Semantic Coherence",
    value=0.5,
    help="Mengukur konsistensi makna antar kalimat. Nilai tinggi menunjukkan isi teks lebih runtut dan saling berhubungan."
)

lexical_diversity_ratio = st.sidebar.number_input(
    "Lexical Diversity",
    value=0.5,
    help="Mengukur keragaman kosakata yang digunakan. Nilai tinggi berarti lebih banyak variasi kata yang muncul dalam teks."
)

readability_grade_level = st.sidebar.number_input(
    "Readability Grade",
    value=10.0,
    help="Menunjukkan tingkat keterbacaan teks berdasarkan level pendidikan. Semakin tinggi nilainya, semakin sulit teks untuk dipahami."
)

generation_confidence_score = st.sidebar.number_input(
    "Generation Confidence",
    value=0.5,
    help="Skor yang menunjukkan tingkat keyakinan bahwa teks dihasilkan oleh model AI. Nilai lebih tinggi mengindikasikan kemungkinan lebih besar berasal dari AI."
)

# ==========================
# MAIN PAGE
# ==========================
st.title("🤖 AI vs Human Text Detection")

st.write(
    """
    Aplikasi ini digunakan untuk mendeteksi apakah suatu teks
    ditulis oleh AI atau manusia menggunakan algoritma
    Random Forest.
    """
)

text = st.text_area(
    "Masukkan Teks",
    height=250,
    placeholder="Masukkan teks berbahasa Inggris..."
)

# ==========================
# PREDIKSI
# ==========================
if st.button("🔍 Prediksi", use_container_width=True):

    if text.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")

    else:

        # ==========================
        # PREPROCESSING
        # ==========================
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

        # ==========================
        # PREDIKSI
        # ==========================
        pred = model.predict(final_input)

        prob = model.predict_proba(final_input)

        label = le.inverse_transform(pred)[0]

        confidence = np.max(prob) * 100

        # ==========================
        # HASIL PREDIKSI
        # ==========================
        st.subheader("📋 Hasil Prediksi")

        if label.lower() == "ai":
            st.error(f"🤖 Hasil Prediksi : {label}")
        else:
            st.success(f"👤 Hasil Prediksi : {label}")

        st.info(
            f"🎯 Tingkat Keyakinan Model : {confidence:.2f}%"
        )

        st.progress(float(confidence) / 100)

        # ==========================
        # ANALISIS HASIL KLASIFIKASI
        # ==========================
        st.subheader("📈 Analisis Hasil Klasifikasi")

        prob_df = pd.DataFrame({
            "Kelas": le.classes_,
            "Probabilitas (%)": prob[0] * 100
        })

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            prob_df["Kelas"],
            prob_df["Probabilitas (%)"]
        )

        ax.set_title(
            "Distribusi Probabilitas Prediksi"
        )

        ax.set_xlabel("Kelas")

        ax.set_ylabel("Probabilitas (%)")

        st.pyplot(fig)

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        # ==========================
        # FEATURE IMPORTANCE
        # ==========================
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

        fig2, ax2 = plt.subplots(
            figsize=(8, 5)
        )

        ax2.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax2.set_title(
            "Feature Importance"
        )

        ax2.set_xlabel(
            "Importance Score"
        )

        st.pyplot(fig2)

        st.dataframe(
            importance_df.sort_values(
                by="Importance",
                ascending=False
            ),
            use_container_width=True
        )

        # ==========================
        # RINGKASAN HASIL
        # ==========================
        st.subheader("📑 Ringkasan Hasil")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediksi",
                label
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )
