#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\loggz\Downloads\mock_sourcing_dataset.xlsx")

df.head()


# Conhecendo e explorando a base

# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.isna().sum().sort_values(ascending=False)


# In[8]:


df.describe(include="all")


# In[9]:


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

df.columns


# In[17]:


date_cols = [
    "sourced_date",
    "first_contact_date",
    "screening_date",
    "interview1_date",
    "test_date",
    "offer_date",
    "hired_date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")


# In[18]:


df.duplicated().sum()


# In[19]:


df["candidate_id"].duplicated().sum()


# In[20]:


df[df["candidate_id"].duplicated(keep=False)]


# #Criando métricas de tempo

# In[29]:


df["contacted"] = df["first_contact_date"].notna()
df["responded"] = df["response_received"] == True
df["passed_screening"] = df["screening_pass"] == True
df["passed_interview"] = df["interview1_pass"] == True
df["took_test"] = df["test_taken"] == True
df["received_offer"] = df["offer_sent"] == True
df["was_hired"] = df["hired"] == True


# #Análise do funil

# In[30]:


funnel = pd.DataFrame({
    "stage": [
        "Sourced",
        "Contacted",
        "Responded",
        "Passed Screening",
        "Passed Interview",
        "Took Test",
        "Received Offer",
        "Hired"
    ],
    "candidates": [
        len(df),
        df["contacted"].sum(),
        df["responded"].sum(),
        df["passed_screening"].sum(),
        df["passed_interview"].sum(),
        df["took_test"].sum(),
        df["received_offer"].sum(),
        df["was_hired"].sum()
    ]
})

funnel["conversion_from_total"] = funnel["candidates"] / len(df)
funnel["conversion_from_previous_stage"] = funnel["candidates"] / funnel["candidates"].shift(1)

funnel


# Conversão por canal

# In[31]:


channel_analysis = (
    df.groupby("source_channel")
    .agg(
        candidates=("candidate_id", "count"),
        responded=("responded", "sum"),
        passed_screening=("passed_screening", "sum"),
        received_offer=("received_offer", "sum"),
        hired=("was_hired", "sum"),
        avg_technical_score=("technical_test_score", "mean"),
        avg_behavior_score=("behavior_score", "mean"),
        avg_manager_score=("manager_score", "mean"),
        avg_response_time=("response_time_days", "mean")
    )
    .reset_index()
)

channel_analysis["response_rate"] = channel_analysis["responded"] / channel_analysis["candidates"]
channel_analysis["offer_rate"] = channel_analysis["received_offer"] / channel_analysis["candidates"]
channel_analysis["hire_rate"] = channel_analysis["hired"] / channel_analysis["candidates"]

channel_analysis.sort_values("hire_rate", ascending=False)


# Conversão por recrutador

# In[32]:


recruiter_analysis = (
    df.groupby("recruiter")
    .agg(
        candidates=("candidate_id", "count"),
        responded=("responded", "sum"),
        passed_screening=("passed_screening", "sum"),
        received_offer=("received_offer", "sum"),
        hired=("was_hired", "sum"),
        avg_stage_duration=("stage_duration_days", "mean")
    )
    .reset_index()
)

recruiter_analysis["hire_rate"] = recruiter_analysis["hired"] / recruiter_analysis["candidates"]
recruiter_analysis["offer_rate"] = recruiter_analysis["received_offer"] / recruiter_analysis["candidates"]

recruiter_analysis.sort_values("hire_rate", ascending=False)


# Comparando contratados x não contratados

# In[33]:


hired_comparison = (
    df.groupby("was_hired")
    .agg(
        avg_years_experience=("years_experience", "mean"),
        avg_response_time=("response_time_days", "mean"),
        avg_technical_score=("technical_test_score", "mean"),
        avg_behavior_score=("behavior_score", "mean"),
        avg_manager_score=("manager_score", "mean"),
        avg_stage_duration=("stage_duration_days", "mean")
    )
)

hired_comparison


# Correlação com contratação

# In[34]:


numeric_cols = [
    "years_experience",
    "response_time_days",
    "technical_test_score",
    "behavior_score",
    "manager_score",
    "stage_duration_days",
    "was_hired"
]

corr = df[numeric_cols].corr(numeric_only=True)

corr["was_hired"].sort_values(ascending=False)


# Criando uma base limpa para exportar e utilizar em visuais

# In[35]:


df_clean = df.copy()

df_clean.to_csv("mock_sourcing_dataset_clean.csv", index=False)
df_clean.to_excel("mock_sourcing_dataset_clean.xlsx", index=False)


# In[37]:


print(funnel)

print(channel_analysis)

print(recruiter_analysis)

print(hired_comparison)


# In[ ]:




