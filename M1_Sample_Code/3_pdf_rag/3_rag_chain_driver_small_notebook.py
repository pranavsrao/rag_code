# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 3. PDF RAG Driver Notebook
# MAGIC
# MAGIC This notebook demonstrates how to use Databricks RAG Studio to log and evaluate a RAG chain with a [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) retrieval component. Note that you will have to first create a vector search endpoint, and a vector search index in order to run this notebook. Please first run the [`3_load_pdf_to_vector_index` notebook]($3_load_pdf_to_vector_index) first to set up this infrastructure. Refer to [the following documentation](https://docs.databricks.com/en/generative-ai/vector-search.html#how-to-set-up-vector-search) for more information on this. 
# MAGIC
# MAGIC This notebook covers the following steps:
# MAGIC
# MAGIC 1. Install required libraries and import required modules
# MAGIC 2. Define paths for the chain notebook and config YAML
# MAGIC 3. Log the chain to MLflow and test it locally, viewing the trace

# COMMAND ----------

# MAGIC %md
# MAGIC ### Establish notebook parameters

# COMMAND ----------

dbutils.widgets.text("rag_chain_config_yaml",   "", label="RAG Chain Config YAML")
dbutils.widgets.text("rag_chain_notebook_path", "", label="RAG Chain Notebook Path")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports and configuration

# COMMAND ----------

import json
import os

import mlflow
from databricks import rag, rag_eval, rag_studio

import html

mlflow.set_registry_uri('databricks-uc')

### START: Ignore this code, temporary workarounds given the Private Preview state of the product
from mlflow.utils import databricks_utils as du
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper functions

# COMMAND ----------

def parse_deployment_info(deployment_info):
  browser_url = du.get_browser_hostname()
  message = f"""Deployment of {deployment_info.model_name} version {deployment_info.model_version} initiated.  This can take up to 15 minutes and the Review App & REST API will not work until this deployment finishes. 

  View status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}
  Review App: {deployment_info.rag_app_url}"""
  return message
### END: Ignore this code, temporary workarounds given the Private Preview state of the product

# COMMAND ----------

# MAGIC %run ../RAG_Experimental_Code

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define paths for chain notebook and config YAML

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook & config YAML
chain_config_file   = dbutils.widgets.get("rag_chain_config_yaml")
chain_notebook_file = dbutils.widgets.get("rag_chain_notebook_path")

chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)
chain_config_path   = os.path.join(os.getcwd(), chain_config_file)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain config path: {chain_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Log the chain
# MAGIC Log the chain to the Notebook's MLflow Experiment inside a Run. The model is logged to the Notebook's MLflow Experiment as a run.

# COMMAND ----------

# DBTITLE 1,Log the model
############
# Log the chain to the Notebook's MLflow Experiment inside a Run
# The model is logged to the Notebook's MLflow Experiment as a run
############

logged_chain_info = rag_studio.log_model(code_path=chain_notebook_path, config_path=chain_config_path)

# Optionally, tag the run to save any additional metadata
with mlflow.start_run(run_id=logged_chain_info.run_id):
  mlflow.set_tag(key="your_custom_tag", value="info_about_chain")

# Save YAML config params to the Run for easy filtering / comparison later(requires experimental import)
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
RagConfig(chain_config_path).experimental_log_to_mlflow_run(run_id=logged_chain_info.run_id)

print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

############
# If you see this error, go to your chain code and comment out all usage of `dbutils`
############
# ValueError: The file specified by 'code_path' uses 'dbutils' command which are not supported in a chain model. To ensure your code functions correctly, remove or comment out usage of 'dbutils' command.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Return the model URI from this notebook

# COMMAND ----------

dbutils.notebook.exit(logged_chain_info.model_uri)
