# Databricks notebook source
# MAGIC %md ### Install wheel files required for using RAG Studio:

# COMMAND ----------

# DBTITLE 1,Fetch Secrets
rag_studio        = dbutils.secrets.get("kv-llmops-framework", "rag-studio-wheel-link")
rag_eval          = dbutils.secrets.get("kv-llmops-framework", "rag-eval-wheel-link")

mlflow_whl        = dbutils.secrets.get("kv-llmops-framework", "mlflow-wheel")
mlflow_skinny_whl = dbutils.secrets.get("kv-llmops-framework", "mlflow-skinny-wheel")

# COMMAND ----------

# DBTITLE 1,Install RAG Studio Wheels
# MAGIC %pip install $rag_studio
# MAGIC %pip install $rag_eval

# COMMAND ----------

# DBTITLE 1,Install External Dependencies and Update Mlflow
# MAGIC %pip install opentelemetry-api opentelemetry-sdk langchain -U -q
# MAGIC %pip uninstall mlflow mlflow-skinny -y # uninstall existing mlflow to avoid installation issues
# MAGIC
# MAGIC %pip install $mlflow_whl -U
# MAGIC %pip install $mlflow_skinny_whl -U
