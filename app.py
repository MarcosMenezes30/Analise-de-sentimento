from pathlib import Path
import re
import string

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentimento_pipeline.joblib"


def limpar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


@st.cache_resource

def carregar_modelo():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Análise de Sentimento", page_icon="💬")
st.title("💬 Análise de sentimento em português")
st.write("Digite um texto e veja a classificação prevista pelo modelo.")

texto = st.text_area("Texto", height=180, placeholder="Ex.: O atendimento foi rápido e eficiente.")

if st.button("Classificar"):
    if not texto.strip():
        st.warning("Digite algum texto antes de classificar.")
    else:
        modelo = carregar_modelo()
        texto_limpo = limpar_texto(texto)
        pred = modelo.predict([texto_limpo])[0]
        proba = modelo.predict_proba([texto_limpo])[0]
        classes = modelo.classes_
        distribuicao = {classe: float(prob) for classe, prob in zip(classes, proba)}

        st.subheader("Resultado")
        st.success(f"Classe prevista: **{pred}**")
        st.write("Probabilidades:")
        st.json(distribuicao)