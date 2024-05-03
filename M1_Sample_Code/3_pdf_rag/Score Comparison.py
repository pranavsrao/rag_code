# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog development

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

df = spark.table('development.rag_studio.rate_case_eval_set_assessments')
display(df)

# COMMAND ----------

pdf = df.toPandas()

# COMMAND ----------

pdf = pdf.drop(columns=['source_type', 'source_id', 'source_metadata', 'retrieval_assessment'])

# COMMAND ----------

pd.set_option('display.max_colwidth', None)

# COMMAND ----------

factual_accuracy_scores = [2,2,2,4,5,
                           0,0,0,4,
                           0,0,0,3,
                           4,4,4,4,3,
                           5,5,5,
                           3,3,3,4,
                           0,0,0,5
                           ]

# COMMAND ----------

fixed_factual_accuracy_scores = [1 if item == 0 else item for item in factual_accuracy_scores]
fixed_factual_accuracy_scores

# COMMAND ----------

df1 = pdf[pdf['app_version'] == '1b']
df2 = pdf[pdf['app_version'] == '2b']
dfs = [df1, df2]

for i in dfs:
    answer_good = []
    metrics = i['response_assessment']
    for metric in metrics:
        answer_good.append(metric['ratings']['answer_good']['double_value'])
    i['answer_good'] = answer_good

df1 = df1.drop(columns=['app_version', 'response_assessment'])
df1 = df1.rename(columns = {'answer_good': 'no_shot_answer_good'})
df1 = df1.sort_values('request_id')
df1 = df1.reset_index(drop=True)
df1['human_score'] = fixed_factual_accuracy_scores

df2 = df2.drop(columns=['app_version', 'response_assessment'])
df2 = df2.rename(columns = {'answer_good': 'few_shot_answer_good'})
df2 = df2.sort_values('request_id')
df2 = df2.reset_index(drop=True)

# COMMAND ----------

df2


# COMMAND ----------

final_df = pd.merge(df1, df2, on='request_id',  how='left')

# COMMAND ----------

no_shot_squared_error = (final_df['no_shot_answer_good']-final_df['human_score'])**2
few_shot_squared_error = (final_df['few_shot_answer_good']-final_df['human_score'])**2

# COMMAND ----------

final_df['no_shot_squared_error'] = no_shot_squared_error
final_df['few_shot_squared_error'] = few_shot_squared_error
final_df

# COMMAND ----------

summary = pd.DataFrame(
    {
        'no_shot_rmse': (final_df['no_shot_squared_error'].mean())**0.5,
        'few_shot_rmse': (final_df['few_shot_squared_error'].mean())**0.5
    },
    index =[0]
)

summary

# COMMAND ----------


