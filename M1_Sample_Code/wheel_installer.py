# Databricks notebook source
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/databricks-rag-studio/679d2f69-6d26-4340-b301-319a955c3ebd/databricks_rag_studio-0.0.0a2-py3-none-any.whl"
# MAGIC
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/rag-eval/releases/databricks_rag_eval-0.0.0a2-py3-none-any.whl"

# COMMAND ----------

# External dependencies
%pip install opentelemetry-api opentelemetry-sdk langchain -U -q
%pip uninstall mlflow mlflow-skinny -y # uninstall existing mlflow to avoid installation issues

%pip install "https://mlflow-snapshots.s3-us-west-2.amazonaws.com/mlflow-2.11.3-0.fd95310a50de7df2-py3-none-any.whl" -U

# COMMAND ----------

# MAGIC %pip install "https://mlflow-snapshots.s3-us-west-2.amazonaws.com/mlflow_skinny-2.11.3-0.fd95310a50de7df2-py3-none-any.whl" -U

# COMMAND ----------

# MAGIC %pip install "https://mlflow-snapshots.s3-us-west-2.amazonaws.com/mlflow_skinny-2.11.3-0.fd95310a50de7df2-py3-none-any.whl" -U
