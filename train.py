import re # Biblioteca com funcções úteis para limpeza do texto
import string # Biblioteca com funcções úteis para limpeza do texto

from pathlib import Path # Biblioteca para pegar o caminho de arquivos
import joblib # Biblioteca para salvar e ler arquivos
import pandas as pd # Biblioteca para abrir/manipular dados/tabelas

# A biblioteca Sklearn é uma biblioteca específica para ML com diversas funções úteis
from sklearn.feature_extraction.text import TfidfVectorizer # Transforma textos em números (vetorização)
from sklearn.linear_model import LogisticRegression # TIpo de modelo de IA para classificação
from sklearn.metrics import classification_report, accuracy_score # Avaliam o nível de acerto do modelo
from sklearn.model_selection import train_test_split # Separa os dados dois grupos, treino e teste
from sklearn.pipeline import Pipeline # Garante que, além da exitência de etapas, sejam executadas na ordem correta 

# Da linha 16 a 19 está acontecendo, basicamente, um armazenamento dos caminhos até arquivos que serão utilizados
# Isso não é obrigatório, mas é uma boa prática fundamental que mantém a orbanização do código
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sentimentos.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentimento_pipeline.joblib"

# Função com o objetivo de fazer um passo FUNDAMENTAL do PRÉ-PROCESSAMENTO dos dados, a limpeza.
# Para isso estão sendo utilizadas diversas funções das bibliotecas re e string.
# Existem limpezas que vão de caso a caso, ou seja, em outras situações algumas coisas feitas aqui não aconteceriam.
def limpar_texto(texto: str) -> str:
    texto = texto.lower().strip() # Deixa tudo minúsculo 
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto) # Remove links da internet (importante para ESSE caso)
    texto = re.sub(r"\d+", " ", texto) # Remove números (importante para ESSE caso)
    texto = texto.translate(str.maketrans("", "", string.punctuation)) # Remove pontuação 
    texto = re.sub(r"\s+", " ", texto) # Troca espaços seguidos por apenas 1 e depois remove espaços do começo/final
    return texto.strip() # Retorna o texto limpo

# Função principal onde tudo acontece. Aqui começa o PROCESSAMENTO
def main() -> None:
    df = pd.read_csv(DATA_PATH) # Utilização da função read_csv do pandas para ler os textos e rótulos da planilha
    df["texto_limpo"] = df["texto"].astype(str).apply(limpar_texto) #  Cria uma nova coluna com o texto tratado

    # Aqui é onde definimos quantos % dos dados serão destinados a testes ou treinamento
    X_train, X_test, y_train, y_test = train_test_split( # X é a coluna de textos e Y é a coluna de rótulos
        df["texto_limpo"],
        df["rotulo"],
        test_size=0.25, # Aqui foi definido que 25% dos dados serão para teste
        random_state=42, # Fundamental para garantir que, mesmo sendo um ou mais modelos, o treinamento tenha repetibilidade 
        stratify=df["rotulo"],
    )

    # Aqui é onde montaremos o pipeline, ou seja, definimos o que será feito e a ordem
    pipeline = Pipeline(
        steps=[
            
            (
                "tfidf", # Apelido dessa etapa
                
                TfidfVectorizer( # Processo de transformação de texto em números e armazenando-os num vetor
                                 # Etapa fundamental uma vez que uma IA entede um texto apenas como uma string
                    
                    ngram_range=(1, 2), # Possibilita trabalhar tanto com palavras simples quanto compostas
                    
                    min_df=1, # Só ignora palavras que apareçam em menos de um documento, ou seja, nenhuma
                    
                    max_df=0.95, # Ignora palavras que apareçam mais de 95% dos textos. Comumente as stopwords
                    
                    sublinear_tf=True, # Define que palavras que menos aparecem tem mais peso
                ),
            ),
            
            (
                "clf", # Apelido dessa etapa
                
                LogisticRegression( # Algoritmo de IA que recebe os números da etapa anterios e aprende com eles
                    
                    max_iter=1000, # Quantas vezes o modelo vai ajustar os pesos dos números da etapa anterior para tentar chegar no melhor resultado
                    
                    class_weight="balanced", # Ajuste feito para efitar respostas enviesada
                    
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train) # Momento em que definimos quais dados préviamente separados vão ser usados para treino
    y_pred = pipeline.predict(X_test) # Pede para a IA adivinhar o sentimento dos textos de teste, ou seja, novos dados

    # Aqui é onde o nível de precisão de acerto da IA é medido
    print("Acurácia:", round(accuracy_score(y_test, y_pred), 4)) # Compara as respostas verdadeiras, y_test, com os palpites da IA, y_pred.
    print("\nRelatório:\n")
    print(classification_report(y_test, y_pred)) # Mostra um relatório detalhado do desempenho incluindo: Quantiade de acertos, precisão, recall, etc

    # Aqui é onde o modelo TREINADO é salvo, garantindo que seja possível reutilizá-lo sem a necessidade de um novo treinamento
    MODEL_DIR.mkdir(exist_ok=True) # Cria uma pasta para salvar o modelo caso não exista
    joblib.dump(pipeline, MODEL_PATH) # Grava o pipeline em um arquivo
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
