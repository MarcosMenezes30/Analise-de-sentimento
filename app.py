# Esse arquivo cria uma página web simples utilizando o steamlist
# O usuário pode digitar um texto qualquer e o modelo de IA dirá qual é o sentimento previsto para esse texto

from pathlib import Path # Facilitador para encontrar e utilizar o caminho de arquivos 
import re # Limpeza e tratamento de texto
import string # Limpeza e tratamento de texto

import joblib # Carregamento de grandes arquivos Python ou modelos já salvos
import streamlit as st # Criação de aplicações wev simples

BASE_DIR = Path(__file__).resolve().parent # A pasta do arquivo atual
MODEL_PATH = BASE_DIR / "models" / "sentimento_pipeline.joblib" # Local onde o arquico do modelo treinado está

# Exerce a MESMA FUNÇÃO em train.py, porém, dessa vez o texto a ser limpo será o do USUÁRIO
def limpar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

# Aqui ocorrerá o carregamento do modelo treinado.
# O decorador da linha 25 faz com que essa operação seja feita só uma vez na vida do app, então o carregamento do modelo só acontece uma vez
@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)

# Apenas ajustando a aparência da página
st.set_page_config(page_title="Análise de Sentimento", page_icon="💬")
st.title("💬 Análise de sentimento em português")
st.write("Digite um texto e veja a classificação prevista pelo modelo.")

# Aqui será o campo onde o usuário digitará o texto a ser analisado e armazena na variável "texto"
texto = st.text_area("Texto", height=180, placeholder="Ex.: O atendimento foi rápido e eficiente.")

# Criação do botão que irá iniciar todo o processo
if st.button("Classificar"):
    if not texto.strip(): # Verificação simples para garantir que algo foi digitado
        st.warning("Digite algum texto antes de classificar.")
    else:
        modelo = carregar_modelo() # Carrega o modelo treinado
        texto_limpo = limpar_texto(texto) # Limpa o texto inserido pelo usuário
        pred = modelo.predict([texto_limpo])[0] # Faz o modelo prever o sentimento daquele texto
        proba = modelo.predict_proba([texto_limpo])[0] # Faz o modelo calcular a probabilidade de cada classe para esse texto
        classes = modelo.classes_ # Oega a lista de nomes das classes que o modelo conhece (positivo, negativo, etc)
        distribuicao = {classe: float(prob) for classe, prob in zip(classes, proba)} # Junta os nomes das classes com as probabilidades em um dicionário Python (Apenas para organização do codigo)

        # Utiliza a biblioteca streamlist para mostrar os resultados na página web
        st.subheader("Resultado")
        st.success(f"Classe prevista: **{pred}**") # Mostra o resultado previsto (posivito, negativo, etc)
        st.write("Probabilidades:")
        st.json(distribuicao) # Mostra as probabilidades de cada classe
