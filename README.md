# Sistema de Análise de Sentimento em Português com NLP e Machine Learning

Projeto de Inteligência Artificial focado na implementação de um sistema de classificação de sentimentos em textos em português utilizando técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning.

Criado para fins acadêmicos do Clube de Inteligência Artificial e Ciência de Dados do Senai Cimatec, o sistema simula um cenário real de análise automatizada de feedback textual, permitindo classificar textos como positivos, negativos ou neutros.

---

## 📝 Licença

Este projeto é fornecido como está para fins educacionais e de demonstração.

- Desenvolvido por: Marcos Menezes  
- Data de Criação: Novembro de 2025  
- Data de Postagem no GitHub: Abril 2026  

---

## 🎯 Objetivo

Simular um sistema de IA aplicado a cenários reais onde grandes volumes de texto precisam ser analisados automaticamente, como:

- avaliações de clientes  
- comentários em redes sociais  
- feedbacks de produtos ou serviços  
- análise de satisfação do usuário  

---

## ⚙️ Tecnologias utilizadas

- Python  
- pandas  
- scikit-learn  
- TF-IDF (vetorização de texto)  
- Logistic Regression  
- Streamlit  

---

## 🧠 Como funciona o sistema

O pipeline segue as etapas clássicas de um sistema de classificação de texto:

1. Coleta de dados  
   Um conjunto de textos rotulados (positivo, negativo e neutro) é utilizado como base.

2. Pré-processamento textual  
   Limpeza dos dados incluindo normalização, remoção de pontuação, números e redução de ruído.

3. Vetorização com TF-IDF  
   Os textos são convertidos em representações numéricas que capturam a importância das palavras.

4. Treinamento do modelo  
   Utilização de algoritmos de Machine Learning (Logistic Regression) para aprender padrões nos dados.

5. Predição  
   O modelo classifica novos textos como positivo, negativo ou neutro.

6. Interface interativa  
   O usuário pode inserir textos em tempo real e visualizar a classificação e probabilidades.

---


## 📊 Diferenciais do projeto

- Implementação completa de pipeline de NLP  
- Uso de técnicas clássas de Machine Learning aplicadas a texto  
- Pré-processamento estruturado de dados textuais  
- Vetorização com TF-IDF  
- Interface interativa com Streamlit  
- Exibição de probabilidades das classes previstas  

---

## ⚠️ Limitações

- Dataset reduzido (não representa todos os contextos reais)  
- Modelo baseado em técnicas clássicas (não utiliza deep learning ou transformers)  
- Sensível à qualidade e variedade dos dados de entrada  

---

## 🚀 Como executar

pip install -r requirements.txt  
python train.py  
streamlit run app.py  
