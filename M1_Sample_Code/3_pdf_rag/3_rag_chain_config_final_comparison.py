# Databricks notebook source
# MAGIC %md
# MAGIC ### Compare two different configurations of LLM-as-a-judge on a pre-graded answer sheet
# MAGIC
# MAGIC Databricks provides a set of metrics that enable you to measure the quality, cost and latency of your RAG app. These metrics are curated by Databricks' Research team as the most relevant (no pun intended) metrics for evaluating RAG applications.
# MAGIC
# MAGIC RAG metrics can be computed using either:
# MAGIC 1. Human-labeled ground truth assessments
# MAGIC 2. LLM judge-labeled assessments 
# MAGIC
# MAGIC A subset of the metrics work only with *either* LLM judge-labeled OR human-labeled ground truth asessments.
# MAGIC
# MAGIC ### Improve judge accuracy
# MAGIC
# MAGIC To improve the accuracy of the Databricks judges, you can provide few-shot examples of "good" and "bad" answers for each LLM judge.  Databricks strongly reccomends providing at least 2 postive and 2 negative examples per judge to improve the accuracy.  See the bottom of the notebook [`5_evaluation_without_rag_studio`](M1_Sample_Code/5_evaluation_without_rag_studio.py) for how to do this.  *Note: Even though this example configuration is included in the non-RAG Studio evaluation example, you can use the example configuration with this notebook.*
# MAGIC
# MAGIC
# MAGIC ### Unstructured docs retrieval & generation metrics
# MAGIC
# MAGIC #### Retriever
# MAGIC
# MAGIC RAG Studio supports the following metrics for evaluating the retriever.
# MAGIC
# MAGIC | Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments & judge name | 
# MAGIC |-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
# MAGIC | Are the retrieved chunks relevant to the user’s query?                            | Precision of "relevant chunk" @ K | 0 to 100% | 0 to 100% | ✔️ | ✔️ `context_relevant_to_question` |
# MAGIC | Are **ALL** chunks that are relevant to the user’s query retrieved?               | Recall of "relevant chunk" @ K | 0 to 100% |0 to 100% | ✔️ |✖️ |
# MAGIC | Are the retrieved chunks returned in the correct order of most to least relevant? | nDCG of "relevant chunk" @ K | 0 to 1 | 0 to 1 |✔️ | ✖️ |
# MAGIC
# MAGIC #### Generation model
# MAGIC
# MAGIC These metrics measure the generation model's performance when the prompt is augemented with unstructured docs from a retrieval step.
# MAGIC
# MAGIC | Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments & judge name | 
# MAGIC |-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
# MAGIC | Is the LLM not hallucinating & responding based ONLY on the context provided? | Faithfulness (to context) | true/false | 0 to 100% | ✖️ | ✔️ `faithful_to_context` |
# MAGIC | Is the response on-topic given the query AND retrieved contexts? | Answer relevance (to query given the context) | true/false | 0 to 100% | ✖️ | ✔️ `relevant_to_question_and_context` | 
# MAGIC | Is the response on-topic given the query? | Answer relevance (to query) | true/false | 0 to 100% | ✖️ | ✔️ `relevant_to_question` | 
# MAGIC | What is the cost of the generation? | Token Count | sum(tokens) | sum(tokens) | n/a |n/a |
# MAGIC | What is the latency of generation? | Latency | milliseconds | average(milliseconds) | n/a | n/a |
# MAGIC
# MAGIC #### RAG chain metrics
# MAGIC
# MAGIC These metrics measure the chain's final response back to the user.  
# MAGIC
# MAGIC | Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments & judge name | 
# MAGIC |-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
# MAGIC | Is the response accurate (correct)? | Answer correctness (vs. ground truth) | true/false | 0 to 100% |✔️ `answer_good` | ✖️ |
# MAGIC | Does the response violate any of my company policies (racism, toxicity, etc)? | Toxicity | true/false | 0 to 100% | ✖️ | ✔️ `harmful` |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies

# COMMAND ----------

# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ../wheel_installer

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Imports and configuration:

# COMMAND ----------

import os
import json
import html

import pandas as pd
import pyspark.sql.functions as F

import mlflow
from databricks import rag, rag_eval, rag_studio

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog development;
# MAGIC use database rag_studio;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Utility and experimental functions:

# COMMAND ----------

### START: Ignore this code, temporary workarounds given the Private Preview state of the product
from mlflow.utils import databricks_utils as du
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "false"

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
# MAGIC ### Load pre-graded answer sheet
# MAGIC
# MAGIC The evaluation set represents the human-annotated ground truth data.
# MAGIC
# MAGIC | Column Name                  | Type                                              | Required? | Comment                                                                                                                                                  |
# MAGIC |------------------------------|---------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | request_id                   | STRING                                            | Either `request_id` or `request` is required        | Id of the request (question)                                                                                                                             |
# MAGIC | request                     | STRING                                            |   Either `request_id` or `request` is required        | A request (question) to the RAG app, e.g., “What is Spark?”                                                                                              |
# MAGIC | expected_response            | STRING                                            |           | (Optional) The expected answer to this question                                                                                                          |
# MAGIC | expected_retrieval_context   | ARRAY<STRUCT<doc_uri: STRING, content: STRING>>   |           | (Optional) The expected retrieval context. The entries are ordered in descending rank. Each entry can record the URI of the retrieved doc and optionally the (sub)content that was retrieved. |
# MAGIC
# MAGIC The responses to these questions can be provided either through an answer sheet (`answer_sheet_table_name`) or they can be generated by calling into an existing model (`model_uri`). Exactly one of these two parameters must be specified.
# MAGIC
# MAGIC If an answer sheet is used, then the schema of `answer_sheet_table_name` is expected to be as follows:
# MAGIC - `request_id`
# MAGIC - `app_version`
# MAGIC - (optional) `response`
# MAGIC - (optional) `retrieval_context`

# COMMAND ----------

############
# Turn the eval dataset into a Delta Table
############
# TODO: Change these values to your catalog and schema
uc_catalog          = "development"
uc_schema           = "rag_studio"

raw_benchmark_table = "bronze_grc_benchmark"
eval_table_name     = "gold_config_test_benchmark"

raw_benchmark_table_name = f"{uc_catalog}.{uc_schema}.{raw_benchmark_table}"
eval_table_fqdn          = f"{uc_catalog}.{uc_schema}.{eval_table_name}"

# COMMAND ----------

sce_benchmark_table = spark.table(raw_benchmark_table_name)

# COMMAND ----------

display(sce_benchmark_table)
# @note: Contains the raw EdisonGPT benchmark data (request_id, request, expected_response, response, factual_accuracy, POC).

# COMMAND ----------

# DBTITLE 1,Create the evaluation benchmark
evaluation_benchmark_table = sce_benchmark_table.select('request_id', 'request', 'expected_response')
# @note: Trimmed benchmark to only include request and expected response. 

display(evaluation_benchmark_table)

# COMMAND ----------

# DBTITLE 1,Create the answer sheet
responses_no_shot = sce_benchmark_table.\
                        select('request_id', 'response').\
                        withColumn('app_version',  F.lit('no_shot'))
# @note: Trimmed to only include the response. Added app version to distinguish metrics from no-shot evaluation configuration from few-shot configuration. 

responses_few_shot = sce_benchmark_table.\
                        select('request_id', 'response').\
                        withColumn('app_version',  F.lit('few_shot'))
# @note: Trimmed to only include the response. Added app version to distinguish metrics from no-shot evaluation configuration from few-shot configuration.

# COMMAND ----------

# DBTITLE 1,Write tables fot Unity Catalog
evaluation_benchmark_table.\
    write.\
    mode('overwrite').\
    saveAsTable('gold_config_test_benchmark')

responses_no_shot.\
    write.\
    mode('overwrite').\
    saveAsTable('gold_config_test_responses_no_shot')

responses_few_shot.\
    write.\
    mode('overwrite').\
    saveAsTable('gold_config_test_responses_few_shot')

# @note: Write tables to Unity Catalog for later use.

# COMMAND ----------

# MAGIC %md ### RunLLM judge-based evaluation configured with no examples

# COMMAND ----------

############
# Turn the eval dataset into a Delta Table
############
response_table_name = 'gold_config_test_responses_no_shot'
response_table_fqdn = f"{uc_catalog}.{uc_schema}.{response_table_name}"

# COMMAND ----------

# DBTITLE 1,YAML Assessment Config
import yaml

config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "relevant_to_question",
                "answer_good"
            ],
        }
    ]
}

config_yml = yaml.dump(config_json)
print(config_yml)

# COMMAND ----------

evaluation_results = rag_eval.evaluate(eval_set_table_name     = eval_table_fqdn,
                                       answer_sheet_table_name = response_table_fqdn, 
                                       config                  = config_yml)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RunLLM judge-based evaluation configured with few-shot examples

# COMMAND ----------


response_table_name = 'gold_config_test_responses_few_shot'
response_table_fqdn = f"{uc_catalog}.{uc_schema}.{response_table_name}"

# COMMAND ----------

# DBTITLE 1,YAML Assessment Config with Few-shot Examples
config_json_2 = {
    "assessment_judges": [
        {
            # "judge_name": "databricks_eval",
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "relevant_to_question",
                {
                    "answer_good": {
                        "examples": [
                            {
                                "request": "What are SCE's wildfire safety measures?",
                                "response": """The document outlines Southern California Edison Company’s (SCE) wildfire safety measures as follows: System Hardening: SCE’s Grid Safety & Resiliency Program (GSRP) and Wildfire Mitigation Program (WMP) include grid resiliency measures, primarily the Wildfire Covered Conductor Program (WCCP), which involves replacing and insulating formerly bare utility wires to reduce ignition risks12. Situational Awareness: SCE is enhancing situational awareness through high-definition cameras and weather stations to detect and monitor potential ignitions, aiming to prevent fire spread3. Inspections and Vegetation Management: SCE has expanded infrastructure inspection programs, such as the Enhanced Overhead Inspection (EOI) program, and increased vegetation management to improve pruning clearances and remove hazardous trees4. Public Outreach and Operational Practices: SCE educates customers about wildfire threats and coordinates with local fire agencies and communities5. During high-risk periods, SCE may proactively de-energize lines through the Public Safety Power Shutoff (PSPS) program6.
                                """,
                                "expected_response": """The document outlines Southern California Edison Company’s (SCE) measures to mitigate wildfires and the associated costs as follows. System Hardening: SCE’s Grid Safety & Resiliency Program (GSRP) and Wildfire Mitigation Program (WMP) include grid resiliency measures like the Wildfire Covered Conductor Program (WCCP) to replace and insulate wires. Situational Awareness: Implementation of high-definition cameras and weather stations to detect and monitor ignitions. Vegetation Management: Expanded programs to increase pruning clearances and remove hazardous trees. Public Outreach and PSPS: Educating customers about wildfire threats and selectively de-energizing lines through the Public Safety Power Shutoff (PSPS) program.""",
                                "value": True,
                                "rationale": "The output details all of the main points covered in the ground truth/expected response",
                            },
                            {
                                "request": "What does SCE stand for?",
                                "response": "South Carolina Electric Company",
                                "expected_response": "Southern California Edison Company",
                                "value": False,
                                "rationale": "The output does not match with the ground truth / expected response.",
                            },
                        ]
                    }
                }
            ],
        }
    ]
}

config_custom_judge_yml = yaml.dump(config_json_2)

# COMMAND ----------

custom_judge_evaluation_results = rag_eval.evaluate(eval_set_table_name=eval_table_fqdn,
                                                    answer_sheet_table_name=response_table_fqdn, 
                                                    config=config_custom_judge_yml)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determine difference in using no-shot vs few-shot examples for configuring LLM-as-a-judge:
# MAGIC
# MAGIC This section is intended to measure whether LLM-judged scores align better to human-judged scores after chaging the evaluation configuration.

# COMMAND ----------

assessments_table_name   = f"{eval_table_fqdn}_assessments"
eval_dataset_assessments = spark.table(assessments_table_name)

display(eval_dataset_assessments)

# COMMAND ----------

eval_dataset_assessments = eval_dataset_assessments.\
                                drop('source_type', 'source_id', 'source_metadata', 'retrieval_assessment').\
                                withColumn('answer_good_score', F.col('response_assessment').ratings.answer_good.double_value).\
                                withColumn('answer_good_justification', F.col('response_assessment').ratings.answer_good.rationale)

display(eval_dataset_assessments)

# COMMAND ----------

factual_accuracy_scores = sce_benchmark_table.\
                            select('request_id', 'factual_accuracy', 'POC').\
                            withColumn('human_score', F.when(F.col('factual_accuracy') == 0, F.lit(1)).otherwise(F.col('factual_accuracy'))).\
                            drop('factual_accuracy')

display(factual_accuracy_scores)

# COMMAND ----------

comparison_table = eval_dataset_assessments.join(factual_accuracy_scores, on=["request_id"])

display(comparison_table)

# COMMAND ----------

display(comparison_table.\
            groupby('app_version').\
            agg( F.count('*').alias('instances'), 
                 F.avg(((F.col('answer_good_score') - F.col('human_score'))**2)).alias('MSE')  ).\
            withColumn('RMSE', F.col('MSE')**0.5) )
            
