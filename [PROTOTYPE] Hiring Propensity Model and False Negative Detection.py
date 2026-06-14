#!/usr/bin/env python
# coding: utf-8

# PT-BR
# Como protótipo prático de IA, desenvolvi um modelo de propensão à contratação utilizando dados históricos de recrutamento. O modelo estima a probabilidade de cada candidato ser contratado com base em resultados de avaliações, características do candidato, canal de sourcing, recrutador responsável, senioridade e área de atuação.
# 
# Além de prever a probabilidade de contratação, o modelo foi utilizado para identificar potenciais falsos negativos: candidatos que foram rejeitados apesar de apresentarem uma alta probabilidade prevista de contratação. Isso cria uma oportunidade para que recrutadores revisem candidatos de alto potencial antes da decisão final de rejeição, aumentando a consistência do processo decisório e reduzindo o risco de deixar passar talentos qualificados.

# EN-US 
# As a practical AI prototype, I developed a hiring propensity model using historical recruitment data. The model estimates each candidate's probability of being hired based on assessment scores, candidate attributes, sourcing channel, recruiter, seniority, and department. 
# 
# In addition to predicting hiring probability, the model was used to identify potential false negatives: candidates who were rejected despite presenting a high predicted probability of being hired. This creates an opportunity for recruiters to review high-potential candidates before final rejection decisions, improving decision consistency and reducing the risk of overlooking strong talent.

# # 1 Hiring Propensity Model

# In[44]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix


# 1) Trazendo a base limpa :)

# In[45]:


df = pd.read_excel(r"C:\Users\loggz\Downloads\mock_sourcing_dataset_clean.xlsx")

df.head()


# 2) Conferindo colunas

# In[46]:


df.columns.tolist()


# 3) Criando variável alvo

# In[47]:


df["target_hired"] = df["hired"].astype(int)


# In[48]:


df["hired"].value_counts(dropna=False)


# 4) Selecionando variáveis do modelo

# In[49]:


features = [
    "source_channel",
    "recruiter",
    "seniority",
    "department",
    "years_experience",
    "response_time_days",
    "technical_test_score",
    "behavior_score",
    "manager_score"
]

target = "target_hired"

model_df = df[features + [target]].copy()


# 5) Tratando valores ausentes

# In[50]:


model_df = model_df.dropna()

model_df.shape


# 6) Separando variáveis numéricas e categóricas

# In[51]:


categorical_features = [
    "source_channel",
    "recruiter",
    "seniority",
    "department"
]

numeric_features = [
    "years_experience",
    "response_time_days",
    "technical_test_score",
    "behavior_score",
    "manager_score"
]


# 7) Separando treino e teste

# In[52]:


X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 8) Criando pipeline do modelo

# In[53]:


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 9) Treinando o modelo

# In[54]:


pipeline.fit(X_train, y_train)


# 10) Avaliando o modelo

# In[55]:


y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 11) Importância das variáveis

# In[56]:


feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importances = pipeline.named_steps["model"].feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

feature_importance.head(15)


# 12) Probabilidade de contratação para todos os candidatos 

# In[57]:


df_model_ready = df.dropna(subset=features).copy()

df_model_ready["hire_probability"] = pipeline.predict_proba(df_model_ready[features])[:, 1]

df_model_ready[[
    "candidate_id",
    "source_channel",
    "recruiter",
    "seniority",
    "department",
    "technical_test_score",
    "behavior_score",
    "manager_score",
    "hired",
    "hire_probability"
]].sort_values("hire_probability", ascending=False).head(20)


# # 2  False negative detection

# 13. Filtrando candidatos rejeitados com alta probabilidade

# In[69]:


false_negative_candidates = df_model_ready[
    (df_model_ready["hired"] == False) &
    (df_model_ready["hire_probability"] >= 0.40)
].copy()


# 14. Candidatos potencialmente overlooked

# In[70]:


false_negative_candidates[[
    "candidate_id",
    "source_channel",
    "recruiter",
    "seniority",
    "department",
    "years_experience",
    "technical_test_score",
    "behavior_score",
    "manager_score",
    "recruiter_notes",
    "rejection_reason",
    "hire_probability"
]].sort_values("hire_probability", ascending=False)


# 15. Criando uma classificação simples

# In[71]:


def classify_candidate_risk(probability):
    if probability >= 0.80:
        return "High potential overlooked"
    elif probability >= 0.70:
        return "Possible false negative"
    else:
        return "Lower priority"

false_negative_candidates["review_flag"] = false_negative_candidates["hire_probability"].apply(classify_candidate_risk)

false_negative_candidates[[
    "candidate_id",
    "review_flag",
    "hire_probability",
    "technical_test_score",
    "behavior_score",
    "manager_score",
    "recruiter_notes",
    "rejection_reason"
]].sort_values("hire_probability", ascending=False)


# 16. Resumo executivo dos falsos negativos

# In[72]:


false_negative_summary = false_negative_candidates.groupby("review_flag").agg(
    candidates=("candidate_id", "count"),
    avg_hire_probability=("hire_probability", "mean"),
    avg_technical_score=("technical_test_score", "mean"),
    avg_behavior_score=("behavior_score", "mean"),
    avg_manager_score=("manager_score", "mean")
).reset_index()

false_negative_summary


# 17. Exportando resultados

# In[73]:


false_negative_candidates.to_excel(
    "false_negative_candidates_review.xlsx",
    index=False
)

feature_importance.to_excel(
    "model_feature_importance.xlsx",
    index=False
)


# # 3 Modelo executivo resumido

# In[75]:


print("="*80)
print("RESUMO EXECUTIVO - MODELO DE PROPENSÃO À CONTRATAÇÃO")
print("="*80)

print("\nDESEMPENHO DO MODELO")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.2%}")

print("\nTOP 10 VARIÁVEIS MAIS IMPORTANTES PARA A CONTRATAÇÃO")

display(
    feature_importance.head(10)
)

print("\nCANDIDATOS DE ALTO POTENCIAL IDENTIFICADOS")

print(
    f"Foram identificados {len(false_negative_candidates)} candidatos rejeitados "
    f"que apresentaram probabilidade prevista de contratação superior a 40%."
)

display(
    false_negative_candidates[
        [
            "candidate_id",
            "hire_probability",
            "technical_test_score",
            "behavior_score",
            "manager_score",
            "source_channel",
            "recruiter"
        ]
    ]
    .sort_values("hire_probability", ascending=False)
    .head(10)
)

print("""
1. O modelo identificou as variáveis mais associadas aos resultados
   de contratação dentro da base analisada.

2. A análise de importância das variáveis permite validar ou desafiar
   os padrões observados na análise exploratória.

3. Candidatos rejeitados com alta probabilidade prevista de contratação,
   quando identificados, podem representar oportunidades de revisão
   das decisões tomadas ao longo do processo.

4. Este protótipo demonstra como IA e Analytics podem apoiar recrutadores
   na tomada de decisão por meio de insights preditivos.
""")


# In[76]:


print("\nINSIGHT EXECUTIVO")

print(
    f"""
O modelo atingiu ROC AUC de {roc_auc_score(y_test, y_proba):.2%},
demonstrando boa capacidade de diferenciar candidatos contratados
e não contratados.

As avaliações técnica, comportamental e gerencial representaram
os fatores mais relevantes para as previsões do modelo,
respondendo por mais de 50% da importância total das variáveis.

Além disso, foi identificado 1 candidato rejeitado com probabilidade
prevista de contratação superior ao limiar definido para revisão,
demonstrando como modelos preditivos podem apoiar recrutadores
na identificação de candidatos que merecem uma segunda análise.
"""
)


# In[79]:


print("\nINSIGHT PARA TALENT")

print(
    f"""
O modelo identificou apenas um candidato rejeitado com probabilidade prevista de contratação superior ao limite 
definido para revisão (40%). Embora isso não signifique necessariamente que a decisão de rejeição tenha sido incorreta, 
o resultado demonstra como análises preditivas podem atuar como um mecanismo complementar de suporte à decisão.

Na prática, esse tipo de modelo permite destacar candidatos que apresentam características semelhantes às de candidatos 
historicamente contratados, possibilitando uma segunda análise antes da decisão final. Neste caso específico, o fato de 
apenas um candidato ter sido sinalizado sugere que, de forma geral, as decisões de contratação observadas na base estão 
alinhadas aos padrões aprendidos pelo modelo, reduzindo evidências de inconsistências ou de potenciais falsos negativos 
no processo seletivo.
"""
)

