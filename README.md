# 🚢 Titanic — Projeto de Machine Learning (Classificação)

Autores: 5MLET — Anderson, Beatriz, Marianna, Murilo, Welder
Data: 2025
Linguagem: Python 3.x
Ambiente: Google Colab / Jupyter Notebook

## 1. Descrição do Projeto

O objetivo deste projeto é prever a probabilidade de sobrevivência dos passageiros do Titanic com base em características demográficas e de viagem, utilizando técnicas de aprendizado supervisionado.

O problema é tratado como uma classificação binária, onde:

0 → passageiro não sobreviveu

1 → passageiro sobreviveu

A base de dados é proveniente do Kaggle Titanic Challenge, amplamente usada em estudos introdutórios de Machine Learning.

## 2. Estrutura do Projeto

```text
.
├── data/
│   ├── train.csv
│   └── test.csv
├── artifacts/
│   └── titanic_best_<modelo>_<timestamp>.joblib
├── tc3_titanic_survival_test_3.py
└── README.md
```
data/: contém os datasets originais do Kaggle.

artifacts/: armazena o pipeline treinado (.joblib).

tc3_titanic_survival_test_3.py: código principal do projeto.

3. Principais Dependências
pip install numpy pandas scikit-learn matplotlib seaborn gradio joblib

Bibliotecas utilizadas:

pandas / numpy: manipulação de dados

matplotlib / seaborn: visualização

scikit-learn: pré-processamento, modelagem e métricas

joblib: serialização de modelos

gradio: criação de interface web interativa

4. Fluxo do Projeto
Etapa 1 — Carregamento dos Dados

Os arquivos train.csv e test.csv devem estar na pasta data/.

df = pd.read_csv("data/train.csv")


O script verifica automaticamente se os arquivos estão disponíveis antes de prosseguir.

Etapa 2 — Análise Exploratória (EDA)

Foram criados diversos gráficos e tabelas para identificar padrões, como:

Distribuição de sobreviventes (Survived)

Taxa de sobrevivência por sexo, classe e porto de embarque

Distribuição de idade entre sobreviventes e não sobreviventes

Relação entre tamanho da família e chance de sobrevivência

Mapa de correlação entre variáveis numéricas

Principais insights:

Mulheres tiveram taxa de sobrevivência significativamente maior.

Passageiros da 1ª classe tiveram melhores chances.

O porto de embarque 'C' apresentou a maior taxa de sobrevivência.

Famílias com 3 pessoas tiveram, em média, as maiores chances de sobrevivência.

Etapa 3 — Pré-processamento

Etapas principais:

Remoção de colunas pouco informativas: PassengerId, Name, Ticket, Cabin

Criação da feature FamilySize = SibSp + Parch + 1

Separação entre features (X) e target (y)

Imputação de nulos:

Numéricas → median

Categóricas → most_frequent

Codificação: OneHotEncoder para variáveis categóricas

Normalização: StandardScaler em variáveis numéricas

Divisão do dataset em train/test (80/20), com stratify=y

Etapa 4 — Modelagem

Modelos treinados:

Logistic Regression

Decision Tree

Random Forest

Todos foram implementados dentro de pipelines (sklearn.pipeline.Pipeline) contendo o pré-processamento + modelo.

Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(...))
])

Etapa 5 — Avaliação

Cada modelo foi avaliado no conjunto de teste com as métricas:

Accuracy

Precision

Recall

F1-score

Matriz de confusão

Os resultados são exibidos em uma tabela comparativa com coloração gradiente (.style.background_gradient).

Etapa 6 — Validação Cruzada (K-Fold)

Uma validação cruzada estratificada (K=5) é executada para avaliar a estabilidade dos modelos.
Métrica principal: F1-score médio e desvio padrão.

Etapa 7 — Interpretabilidade

O modelo com melhor F1 é selecionado.

Se RandomForest: é exibida a importância das features.

Se LogisticRegression: são exibidos os coeficientes dos preditores (sinal e magnitude).

Isso permite identificar as variáveis mais influentes na decisão.

Etapa 8 — Exportação do Modelo

O melhor pipeline (pré-processamento + modelo) é salvo automaticamente em:

artifacts/titanic_best_<Modelo>_<timestamp>.joblib


Assim, ele pode ser reutilizado sem necessidade de reprocessar os dados.

Etapa 9 — Inferência

Um exemplo de predição é demonstrado:

sample = pd.DataFrame([{
    "Pclass": 3, "Sex": "male", "Age": 34,
    "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"
}])


O modelo estima a probabilidade e retorna mensagens como:

✅ Sobreviveria (probabilidade: 83.4%)
❌ Não sobreviveria (probabilidade: 27.1%)

Etapa 10 — Aplicação Web (Gradio)

Uma interface interativa foi criada com Gradio:

O usuário escolhe valores via dropdowns e sliders.

O modelo retorna a previsão com uma barra de probabilidade colorida.

O app pode ser executado localmente ou compartilhado via link público.

Exemplo:

interface.launch(share=True)

5. Métricas Esperadas (exemplo típico)
Modelo	Accuracy	Precision	Recall	F1
RandomForest	0.83	0.81	0.80	0.80
Logistic Regression	0.80	0.78	0.77	0.77
Decision Tree	0.77	0.75	0.74	0.74

(os valores exatos dependem da semente e do dataset local)

6. Conclusões

O Random Forest apresentou o melhor desempenho geral, equilibrando precisão e recall.

As variáveis mais determinantes foram Sexo, Classe (Pclass), Idade, e Tarifa (Fare).

O projeto ilustra o ciclo completo de Machine Learning — da análise inicial à aplicação prática.
