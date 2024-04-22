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
# MAGIC 3. Define paths for the chain notebook and config YAML
# MAGIC 4. Log the chain to MLflow and test it locally, viewing the trace
# MAGIC 5. Evaluate the chain using an eval dataset
# MAGIC 6. Deploy the chain

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies

# COMMAND ----------

# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ../wheel_installer

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

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
# MAGIC # Define paths for chain notebook and config YAML

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook & config YAML
chain_notebook_file = "3_rag_chain"
chain_config_file = "3_rag_chain_config.yaml"

chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)
chain_config_path = os.path.join(os.getcwd(), chain_config_file)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain config path: {chain_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Log the chain
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
# MAGIC # Test the model locally & view the trace

# COMMAND ----------

# DBTITLE 1,Local Model Testing and Tracing
############
# Test the model locally
# This is the same input that the REST API will accept once deployed.
############

model_input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello world!!",
        },
        
    ]
}

loaded_model = mlflow.langchain.load_model(logged_chain_info.model_uri)

# Run the model to see the output
# loaded_model.invoke(question)


############
# Experimental: View the trace
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
json_trace = experimental_get_json_trace(loaded_model, model_input)

json_string = json.dumps(json_trace, indent=4)

# Escape HTML characters to avoid XSS or rendering issues
escaped_json_string = html.escape(json_string)

# Generate HTML with the escaped JSON inside <pre> and <code> tags
pretty_json_html = f"<html><body><pre><code>{escaped_json_string}</code></pre></body></html>"

# To use the HTML string in a context that renders HTML, 
# such as a web application or a notebook cell that supports HTML output
displayHTML(pretty_json_html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the chain

# COMMAND ----------

# MAGIC %md
# MAGIC ## First, build an evaluation set
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

# COMMAND ----------

############
# Experimental: you can query the model to iteratively build your evaluation set
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############

eval_dataset = []
expected_retrieval_context = None
request_id = "sample_1"
request = ""
expected_response = ""

for step in json_trace["steps"]:
  if step['type'] == "RETRIEVAL":
    expected_retrieval_context = step['retrieval']['chunks']
    request = step['retrieval']['query_text']
  elif step['type'] == "LLM_GENERATION":
    expected_response = step['text_generation']['generated_text']

ground_truth_example = {
        "request_id": request_id,
        "request": request,
        # Retrieval context is optional
        "expected_retrieval_context": expected_retrieval_context,
        # Expected response is optional
        "expected_response": expected_response,
    }

ground_truth_example

# COMMAND ----------

# DBTITLE 1,Sample Evaluation Dataset Loader
############
# If you have a known set of queries, you can build the evaluation dataset manually
# Alternatively, you can create the evaluation dataset using Spark/SQL - it is simply an Delta Table with the above schema
############

# eval_dataset = [
#     {
#         "request_id": "sample_request_1",
#         "request": "What is ARES?",
#         # Expected retrieval context is optional, if not provided, RAG Studio will use LLM judge to assess each retrieved context
#         "expected_retrieval_context": [
#             {
#                         "chunk_id": "9517786ecadf3e0c75e3cd4ccefdced5",
#                         "doc_uri": "dbfs:/Volumes/rag/ericp_m1/matei_pdf/2311.09476.pdf",
#                         "content": "..."
                        
#                     },
#                     {
#                         "chunk_id": "e8825fe982f7fd190ad828a307d7f280",
#                         "doc_uri": "dbfs:/Volumes/rag/ericp_m1/matei_pdf/2311.09476.pdf",
#                         "content": "..."
                        
#                     },
#                     {
#                         "chunk_id": "e47b43c9c8f8ce11d78342c49ddbea07",
#                         "doc_uri": "dbfs:/Volumes/rag/ericp_m1/matei_pdf/2311.09476.pdf",
#                         "content": "..."
                        
#                     }
#         ],
#         # Expected response is optional
#         "expected_response": "ARES is an Automated RAG Evaluation System for evaluating retrieval-augmented generation (RAG) systems along the dimensions of context relevance, answer faithfulness, and answer relevance. It uses synthetic training data to finetune lightweight LM judges to assess the quality of individual RAG components and utilizes a small set of human-annotated datapoints for prediction-powered inference (PPI) to mitigate potential prediction errors.",
#     }
# ]

eval_dataset = [
    {
        "request_id": "sample_request_1",
        "request": "What is the purpose of SCE's rate case application?",
        # Expected retrieval context is optional, if not provided, RAG Studio will use LLM judge to assess each retrieved context
        # Expected response is optional
        "expected_response": """The purpose of Southern California Edison Company’s (SCE) rate case application is to request an authorized base revenue requirement (ABRR) of $7.601 billion for the test year 2021, which would reflect an increase in rates to support various objectives. These include maintaining and improving the electric grid, implementing the state’s public policy goals such as wildfire mitigation and de-carbonization through electrification, and adapting to a rapidly modernizing grid. The application also seeks to recover expenditures recorded in various wildfire-related memorandum accounts for 2019 and 20203. Additionally, SCE aims to continue providing safe, reliable, affordable, and increasingly clean electricity to its customers while addressing the challenges posed by the global climate crisis and catastrophic wildfires."""
    },
    {
        "request_id": "sample_request_2",
        "request": "What are different reasons SCE has stated for its request?",
        "expected_response": """The document outlines several reasons for Southern California Edison’s (SCE) request in the General Rate Case Application: Safety and Reliability: SCE emphasizes the need to maintain and improve the grid to ensure safety and reliability for customers, especially in light of the increased risk of catastrophic wildfires.Wildfire Mitigation: The company proposes comprehensive programs and actions aimed at mitigating wildfire risks associated with their equipment due to extreme environmental conditions.Regulatory Compliance: SCE’s request includes necessary expenditures to comply with new regulations and legislative requirements, such as those from Assembly Bill 1054. Clean Energy Goals: SCE is committed to supporting the state’s clean energy policies, including the reduction of greenhouse gas emissions and the integration of distributed energy resources."""
    },
    {
        "request_id": "sample_request_3",
        "request":  "Which regulations affect SCE's expenses?",
        "expected_response": """The document outlines several regulations that impact Southern California Edison Company’s (SCE) expenses: Fire-Threat Safety Map OIR: Regulations for electric power lines in high fire-threat areas, including new inspection, maintenance, and vegetation management regulations. Risk Assessment Mitigation Phase (RAMP): SCE’s RAMP Report examines top safety risks and identifies new mitigations, influencing operational and capital expenditures. Grid Safety & Resiliency Program (GSRP): SCE’s GSRP Application proposes wildfire mitigation programs and activities, impacting costs related to grid hardening. Wildfire Mitigation Plan (WMP): SCE’s WMP describes programs and activities for wildfire risk mitigation, affecting wildfire-related costs. These regulations require SCE to undertake specific actions and programs that contribute to their overall expenses."""
    },
    {
        "request_id": "sample_request_4",
        "request": "Which measures will SCE take to mitigate wildfires?",
        "expected_response": """The document outlines Southern California Edison Company’s (SCE) measures to mitigate wildfires and the associated costs as follows. System Hardening: SCE’s Grid Safety & Resiliency Program (GSRP) and Wildfire Mitigation Program (WMP) include grid resiliency measures like the Wildfire Covered Conductor Program (WCCP) to replace and insulate wires. Situational Awareness: Implementation of high-definition cameras and weather stations to detect and monitor ignitions. Vegetation Management: Expanded programs to increase pruning clearances and remove hazardous trees. Public Outreach and PSPS: Educating customers about wildfire threats and selectively de-energizing lines through the Public Safety Power Shutoff (PSPS) program."""
    },
    {
        "request_id": "sample_request_5",
        "request": "What measures is SCE taking to protect low-income customers from increased electricity costs?",
        "expected_response": """The document outlines SCE’s commitment to affordability, especially for low-income customers, in the context of their General Rate Case (GRC) application. Here are the key points: Affordability Assessment: SCE evaluates the impact of rate increases on all customers, with a particular focus on low-income customers.Four-Part Framework: SCE proposes a framework to assess affordability, which includes evaluating the reasonableness of objectives, scope, pace, and cost, as well as the impact on customers’ ability to pay their bills. Income-Qualified Customers: Special attention is given to income-qualified customers to ensure they are not overburdened by energy expenditures. Policy Compliance: SCE’s efforts align with the state policy that essential electricity should be affordable for all residents. SCE’s approach is to balance the need for infrastructure investments with the importance of keeping electricity affordable for low-income customers."""
    },    
]

############
# Turn the eval dataset into a Delta Table
############
# TODO: Change these values to your catalog and schema
uc_catalog = "development"
uc_schema = "rag_studio"
eval_table_name = "rate_case_eval_set"
eval_table_fqdn = f"{uc_catalog}.{uc_schema}.{eval_table_name}"

df = spark.read.json(spark.sparkContext.parallelize(eval_dataset))
df.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
    eval_table_fqdn
)
print(f"Eval set written to: {eval_table_fqdn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the evaluation
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

# DBTITLE 1,YAML Assessment Config Parser
import yaml
############
# Note the judge names are fixed values per the table above.
############

############
# Default evaluation configuration
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}


############
# Currently, evaluation is slow with the Databricks provided LLM judge due to a limitation we are working to remove.  You can temporarily use any Model Serving endpoint to overcome this limitation, including DBRX.
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}

config_yml = yaml.dump(config_json)
print(config_yml)

# COMMAND ----------

# DBTITLE 1,Machine Learning Experiment Tracker
############
# Run evaluation, logging the results to a sub-run of the chain's MLflow run
############
with mlflow.start_run(logged_chain_info.run_id):
  evaluation_results = rag_eval.evaluate(eval_set_table_name=eval_table_fqdn, model_uri=logged_chain_info.model_uri, config=config_yml)

############
# Experimental: Log evaluation results to MLflow.  Note you can also use the dashboard produced by RAG Studio to view metrics/debug quality - it has more advanced functionality.
# Known issues: Can only be run once per run_id.
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
experimental_add_metrics_to_run(evaluation_results, evaluation_results.mlflow_run_id)
experimental_add_eval_outputs_to_run(evaluation_results, evaluation_results.mlflow_run_id)
experimental_add_eval_config_tags_to_run(evaluation_results, config_yml, evaluation_results.mlflow_run_id)
# Note: If you change the config after you log the model, but before you run this command, the incorrect config will be logged.
RagConfig(chain_config_path).experimental_log_to_mlflow_run(run_id=evaluation_results.mlflow_run_id)


# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy the model to the Review App
# MAGIC
# MAGIC To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.

# COMMAND ----------

# DBTITLE 1,Deploy the model
# TODO: Change these values to your catalog and schema
uc_catalog = "development"
uc_schema = "rag_studio"
model_name = "rate_case_2"
uc_model_fqdn = f"{uc_catalog}.{uc_schema}.{model_name}" 

uc_registered_chain_info = mlflow.register_model(logged_chain_info.model_uri, uc_model_fqdn)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the chain to:
# MAGIC 1. Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# MAGIC 2. Chain REST API endpoint to call the chain from your front end.
# MAGIC 3. Feedback REST API endpoint to pass feedback back from your front end.
# MAGIC
# MAGIC **Note:** It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

deployment_info = rag_studio.deploy_model(uc_model_fqdn, uc_registered_chain_info.version)
print(parse_deployment_info(deployment_info))

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

import inspect

# COMMAND ----------

print(inspect.getsource(rag_studio.deploy_model))

# COMMAND ----------

print(inspect.getsource(rag_studio))

# COMMAND ----------

print(inspect.getsource(databricks.rag_studio.client.rest_client.deploy_chain))

# COMMAND ----------

import mlflow
import uuid

# SDK for rag
from databricks.rag_studio.permissions import (
    _check_manage_permissions_on_deployment,
    _check_view_permissions_on_deployment,
)
from mlflow import set_registry_uri, start_run
from mlflow.utils import databricks_utils
from mlflow.pyfunc import log_model
from typing import List, Optional, Tuple
from databricks.rag_studio.version import VERSION
from databricks.rag_studio.entities import Deployment, PermissionLevel
from databricks.rag_studio.client.rest_client import (
    get_chain_deployments as rest_get_chain_deployments,
    deploy_chain as rest_deploy_chain,
    list_chain_deployments as rest_list_chain_deployments,
    create_review_artifacts as rest_create_review_artifacts,
    get_review_artifacts as rest_get_review_artifacts,
)
from databricks.rag_studio.feedback import DummyFeedbackModel, input_signature
from databricks.rag_studio.utils.mlflow_utils import _get_latest_model_version
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointCoreConfigOutput,
    ServedModelInput,
    TrafficConfig,
    Route,
    AutoCaptureConfigInput,
    ServedModelInputWorkloadSize,
    EndpointPendingConfig,
)
from databricks.sdk.errors.platform import (
    ResourceDoesNotExist,
    ResourceConflict,
    InvalidParameterValue,
    BadRequest,
    Unauthenticated,
    NotFound,
    PermissionDenied,
)
from databricks.sdk.service.serving import (
    ServingEndpointAccessControlRequest,
    ServingEndpointPermissionLevel,
    ServingEndpointPermissions,
)
from databricks.sdk.service.workspace import (
    WorkspaceObjectPermissionLevel,
    WorkspaceObjectPermissions,
    WorkspaceObjectAccessControlRequest,
)
import re

_TRACES_FILE_PATH = "traces.json"
_FEEDBACK_MODEL_NAME = "feedback"


def get_deployments(
    model_name: str, model_version: Optional[int] = None
) -> List[Deployment]:
    deployments = rest_get_chain_deployments(model_name, model_version)
    # TODO(ML-39693): Filter out deleted endpoints
    if len(deployments) > 0:
        _check_view_permissions_on_deployment(deployments[0])
    return deployments


def _create_served_model_input(model_name, version, model_input_name, scale_to_zero):
    return ServedModelInput(
        name=_remove_dots(model_input_name),
        model_name=model_name,
        model_version=version,
        workload_size=ServedModelInputWorkloadSize.SMALL,
        scale_to_zero_enabled=scale_to_zero,
    )


def _check_model_name(model_name):
    if len(model_name.split(".")) != 3:
        raise ValueError("Model name must be in the format catalog.schema.model_name")


def _get_catalog_and_schema(model_name):
    parts = model_name.split(".")
    return parts[0], parts[1]


def _remove_dots(model_name):
    return model_name.replace(".", "-")


def _create_endpoint_name(model_name):
    full_name = f"rag_studio_{_remove_dots(model_name)}"
    return full_name[:60]


def _create_served_model_name(model_name, version):
    full_name = f"{_remove_dots(model_name)}_{version}"
    return full_name[:60]


def _create_feedback_model_name(model_name: str) -> str:
    catalog_name, schema_name = _get_catalog_and_schema(model_name)
    return f"{catalog_name}.{schema_name}.{_FEEDBACK_MODEL_NAME}"


def _create_feedback_model(
    feedback_uc_model_name: str, scale_to_zero: bool
) -> ServedModelInput:
    set_registry_uri("databricks-uc")

    # only create the feedback model if it doesn't already exist in this catalog.schema
    feedback_model_version = str(_get_latest_model_version(feedback_uc_model_name))
    if feedback_model_version == "0":
        with start_run(run_name="feedback-model"):
            # also adds to UC with version '1'
            log_model(
                artifact_path=_FEEDBACK_MODEL_NAME,
                python_model=DummyFeedbackModel(),
                signature=input_signature,
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/databricks-rag-studio/679d2f69-6d26-4340-b301-319a955c3ebd/databricks_rag_studio-{VERSION}-py3-none-any.whl",
                ],
                registered_model_name=feedback_uc_model_name,
            )
            feedback_model_version = str(
                _get_latest_model_version(feedback_uc_model_name)
            )
    return _create_served_model_input(
        feedback_uc_model_name,
        feedback_model_version,
        _FEEDBACK_MODEL_NAME,
        scale_to_zero,
    )


def _parse_pending_config_for_feedback_config(
    uc_model_name: str, pending_config: EndpointPendingConfig
) -> EndpointCoreConfigOutput:
    """
    Parse pending_config to get additional information about the feedback model in order to
    return a config as if the endpoint was successfully deployed with only the feedback model.
    This way we can reuse the update functions that are written for normal endpoint updates.
    """
    feedback_uc_model_name = _create_feedback_model_name(uc_model_name)
    for model in pending_config.served_models:
        if model.name == _FEEDBACK_MODEL_NAME:
            feedback_model_version = model.model_version
            scale_to_zero = model.scale_to_zero_enabled

    return EndpointCoreConfigOutput(
        served_models=[
            _create_served_model_input(
                model_name=feedback_uc_model_name,
                version=feedback_model_version,
                model_input_name=_FEEDBACK_MODEL_NAME,
                scale_to_zero=scale_to_zero,
            )
        ],
        traffic_config=TrafficConfig(
            routes=[Route(served_model_name=_FEEDBACK_MODEL_NAME, traffic_percentage=0)]
        ),
        auto_capture_config=pending_config.auto_capture_config,
    )


def _construct_table_name(catalog_name, schema_name, model_name):
    w = WorkspaceClient()
    # remove catalog and schema from model_name and add rag_studio- prefix
    base_name = "rag_studio-" + model_name.split(".")[2]
    suffix = ""

    # try to append suffix
    for index in range(20):
        if index != 0:
            suffix = f"_{index}"

        table_name = f"{base_name[:63-len(suffix)]}{suffix}"

        full_name = f"{catalog_name}.{schema_name}.{table_name}_payload"
        if not w.tables.exists(full_name=full_name).table_exists:
            return table_name

    # last attempt - append uuid and truncate to 63 characters (max length for table_name_prefix)
    # unlikely to have conflict unless base_name is long
    if len(base_name) > 59:
        return f"{base_name[:59]}_{uuid.uuid4().hex}"[:63]
    return f"{base_name}_{uuid.uuid4().hex}"[:63]


def _create_new_endpoint_config(
    model_name, version, endpoint_name, scale_to_zero=False
):
    catalog_name, schema_name = _get_catalog_and_schema(model_name)

    served_model_name = _create_served_model_name(model_name, version)
    feedback_uc_model_name = _create_feedback_model_name(model_name)

    table_name = _construct_table_name(catalog_name, schema_name, model_name)

    return EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[
            _create_served_model_input(
                model_name, version, served_model_name, scale_to_zero
            ),
            _create_feedback_model(feedback_uc_model_name, scale_to_zero),
        ],
        traffic_config=TrafficConfig(
            routes=[
                Route(
                    served_model_name=served_model_name,
                    traffic_percentage=100,
                ),
                Route(
                    served_model_name=_FEEDBACK_MODEL_NAME,
                    traffic_percentage=0,
                ),
            ]
        ),
        auto_capture_config=AutoCaptureConfigInput(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name_prefix=table_name,
        ),
    )


def _update_traffic_config(
    model_name: str,
    version: str,
    existing_config: EndpointCoreConfigOutput,
) -> TrafficConfig:
    served_model_name = _create_served_model_name(model_name, version)
    updated_routes = [
        Route(served_model_name=served_model_name, traffic_percentage=100)
    ]

    if existing_config:
        for traffic_config in existing_config.traffic_config.routes:
            updated_routes.append(
                Route(
                    served_model_name=traffic_config.served_model_name,
                    traffic_percentage=0,
                )
            )
    return TrafficConfig(routes=updated_routes)


def _update_served_models(
    model_name: str,
    version: str,
    endpoint_name: str,
    existing_config: EndpointCoreConfigOutput,
    scale_to_zero: bool,
) -> List[ServedModelInput]:
    served_model_name = _create_served_model_name(model_name, version)
    updated_served_models = [
        _create_served_model_input(
            model_name, version, served_model_name, scale_to_zero
        )
    ]

    if existing_config:
        updated_served_models.extend(existing_config.served_models)

    return updated_served_models


def _construct_query_endpoint(workspace_url, endpoint_name, model_name, version):
    # This is a temporary solution until we can identify the appropriate solution to get
    # the workspace URI in backend. Ref: https://databricks.atlassian.net/browse/ML-39391
    served_model_name = _create_served_model_name(model_name, version)
    return f"{workspace_url}/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/invocations"


def deploy_model(model_name: str, version: int, **kwargs) -> Deployment:
    _check_model_name(model_name)
    endpoint_name = _create_endpoint_name(model_name)
    w = WorkspaceClient()
    scale_to_zero = kwargs.get("scale_to_zero", False)
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
    except ResourceDoesNotExist:
        w.serving_endpoints.create(
            name=endpoint_name,
            config=_create_new_endpoint_config(
                model_name, version, endpoint_name, scale_to_zero
            ),
        )
    else:
        config = endpoint.config
        # TODO: https://databricks.atlassian.net/browse/ML-39649
        # config=None means this endpoint has never successfully deployed before
        # bc we have a dummy feedback model, we know feedback works, so we only want its config
        if config is None:
            config = _parse_pending_config_for_feedback_config(
                model_name, endpoint.pending_config
            )

        # ignore pending_config bc we only redeploy models that have succesfully deployed before
        # set the traffic config for all currently deployed models to be 0
        updated_traffic_config = _update_traffic_config(model_name, version, config)
        updated_served_models = _update_served_models(
            model_name, version, endpoint_name, config, scale_to_zero
        )
        try:
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_models=updated_served_models,
                traffic_config=updated_traffic_config,
                auto_capture_config=config.auto_capture_config,
            )
        except ResourceConflict:
            raise ValueError("The endpoint is currently updating")
        except InvalidParameterValue as e:
            if "served_models cannot contain more than 15 elements" in str(e):
                raise ValueError(
                    "There are already 15 models deployed to this endpoint. Please delete one before deploying."
                )
            else:
                # pass through any other errors
                raise e
        except BadRequest as e:
            if "Cannot create 2+ served entities" in str(e):
                raise ValueError(
                    """You cannot redeploy the same model and version more than once.
Update the version number"""
                )
            else:
                raise e

    workspace_url = f"https://{databricks_utils.get_browser_hostname()}"
    return rest_deploy_chain(
        model_name=model_name,
        model_version=version,
        query_endpoint=_construct_query_endpoint(
            workspace_url, endpoint_name, model_name, version
        ),
        endpoint_name=endpoint_name,
        served_entity_name=_create_served_model_name(model_name, version),
        workspace_url=workspace_url,
    )


def list_deployments() -> List[Deployment]:
    deployments = rest_list_chain_deployments()
    # call check view permissions for each deployment and filter out if can't view or deleted
    deployments_with_permissions = []
    for deployment in deployments:
        try:
            _check_view_permissions_on_deployment(deployment)
            deployments_with_permissions.append(deployment)
        except ValueError:
            pass
        except ResourceDoesNotExist:
            pass
    return deployments_with_permissions


def _get_run_ids_from_artifact_uris(artifact_uris: List[str]) -> List[str]:
    return [
        re.search(r"runs:/(.*?)/.*", artifact_id).group(1)
        for artifact_id in artifact_uris
    ]


def _get_experiment_ids(run_ids: List[str]) -> List[str]:
    w = WorkspaceClient()
    experiment_ids = set()
    for run_id in run_ids:
        run_response = w.experiments.get_run(run_id)
        experiment_ids.add(run_response.run.info.experiment_id)
    return list(experiment_ids)


def _get_table_name(auto_capture_config):
    catalog_name = auto_capture_config.catalog_name
    schema_name = auto_capture_config.schema_name
    table_name = auto_capture_config.state.payload_table.name
    return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"


def _get_inference_table_from_serving(model_name, serving_endpoint_name):
    w = WorkspaceClient()
    serving_endpoint = w.serving_endpoints.get(serving_endpoint_name)
    if (
        serving_endpoint.config is None
        or serving_endpoint.config.auto_capture_config is None
    ):
        raise ValueError(
            f"The provided {model_name} doesn't have any inference table configured. "
            "Please update the endpoint to capture payloads to an inference table"
        )

    auto_capture_config = serving_endpoint.config.auto_capture_config
    if (
        auto_capture_config.catalog_name is None
        or auto_capture_config.schema_name is None
    ):
        raise ValueError(
            f"The provided {model_name} doesn't have any inference table configured. "
            "Please update the endpoint to capture payloads to an inference table"
        )

    return _get_table_name(auto_capture_config)


def _convert_inference_table_to_tracing_schema(request_payloads):
    """
    Convert the inference table to the schema required for tracing
    """
    from pyspark.sql import functions as F
    from databricks.rag_studio.utils.schemas import (
        MESSAGES_SCHEMA,
        CHOICES_SCHEMA,
        TRACE_SCHEMA,
    )

    changed_request_payloads = request_payloads.filter(
        F.expr("response:choices IS NOT NULL")
    ).withColumn(  # Ignore error requests
        "timestamp", (F.col("timestamp_ms") / 1000).cast("timestamp")
    )

    return (
        changed_request_payloads.withColumn(
            "request",
            F.struct(
                F.col("databricks_request_id").alias("request_id"),
                F.expr("request:databricks_options.conversation_id").alias(
                    "conversation_id"
                ),
                F.col("timestamp"),
                F.from_json(F.expr("request:messages"), MESSAGES_SCHEMA).alias(
                    "messages"
                ),
                F.element_at(
                    F.from_json(F.expr("request:messages"), MESSAGES_SCHEMA), -1
                )
                .getItem("content")
                .alias("last_input"),
            ),
        )
        .withColumn(
            "trace",
            F.from_json(F.expr("response:databricks_output.trace"), TRACE_SCHEMA),
        )
        .withColumn(
            "output",
            F.struct(
                F.from_json(F.expr("response:choices"), CHOICES_SCHEMA).alias("choices")
            ),
        )
        .select("request", "trace", "output")
    )


def enable_trace_reviews(
    model_name: str, request_ids: Optional[List[str]] = None
) -> str:
    """
    This enables the reviewer UI to start capturing feedback from reviewer
    on the request_ids provided for a particular model.

    :param model_name: The name of the UC Registered Model to use when
                registering the chain as a UC Model Version.
                Example: catalog.schema.model_name
    :param request_ids: Optional list of request_ids for which the feedback
                needs to be captured. Example: ["490cf09b-6da6-474f-bc35-ee5ca688ff8d",
                "a4d37810-5cd0-4cbd-aa25-e5ceaf6a448b"]

    :return: The Feedback URL where users can start providing feedback for the request_ids

    Example:
    ```
    from databricks.rag_studio import enable_trace_reviews

    enable_trace_reviews(
        model_name="catalog.schema.chain_model",
        request_ids=["490cf09b-6da6-474f-bc35-ee5ca688ff8d", "a4d37810-5cd0-4cbd-aa25-e5ceaf6a448b"],
    )
    ```
    """
    chain_deployments = rest_get_chain_deployments(model_name)
    [
        _check_manage_permissions_on_deployment(deployment)
        for deployment in chain_deployments
    ]
    if len(chain_deployments) == 0:
        raise ValueError(
            f"The provided {model_name} has never been deployed. "
            "Please deploy the model first using deploy_chain API"
        )
    chain_deployment = chain_deployments[-1]
    serving_endpoint_name = chain_deployment.endpoint_name
    table_full_name = _get_inference_table_from_serving(
        model_name, serving_endpoint_name
    )

    if request_ids:
        # cast id to int if other type is passed in
        request_ids_str = ", ".join([f"'{id}'" for id in request_ids])
        sql_query = f"SELECT * FROM {table_full_name} WHERE databricks_request_id IN ({request_ids_str})"
    else:
        sql_query = f"SELECT * FROM {table_full_name}"

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    try:
        spark_df = spark.sql(sql_query)
        converted_spark_df = _convert_inference_table_to_tracing_schema(spark_df)
        df = converted_spark_df.toPandas()
    except Exception as e:
        raise ValueError(
            f"Failed to fetch the data from the table {table_full_name}. Error: {str(e)}"
        )

    with mlflow.start_run() as model_run:
        mlflow.log_table(data=df, artifact_file=_TRACES_FILE_PATH)
        artifact_uri = f"runs:/{model_run.info.run_id}/{_TRACES_FILE_PATH}"
        rest_create_review_artifacts(model_name, artifacts=[artifact_uri])

    return chain_deployment.rag_app_url


# Given an endpoint, calls it with the appropriate arguments and handles errors
def _call_workspace_api(endpoint, kwargs):
    try:
        return endpoint(**kwargs)
    except Unauthenticated as e:
        raise ValueError(
            "Unable to authenticate to the databricks workspace: " + str(e)
        )
    except PermissionDenied:
        raise ValueError(
            "Permission Denied: User does not have valid permissions for setting permssions on the deployment."
        )
    except NotFound as e:
        raise ValueError(
            "Invalid Inputs: Passed in parameters are not found. " + str(e)
        )
    except ResourceDoesNotExist as e:
        raise ValueError("Resource does not exist, please check your inputs: " + str(e))
    except Exception as e:
        raise e


# Get permissions on a given endpoint
def _get_permissions_on_endpoint(endpoint_id: str) -> ServingEndpointPermissions:
    w = WorkspaceClient()
    permissions = _call_workspace_api(
        w.serving_endpoints.get_permissions, {"serving_endpoint_id": endpoint_id}
    )
    return permissions


# Get permissions on a given experiment
# TODO: Handle a normal experiment as well (ML-39642)
def _get_permissions_on_experiment(experiment_id: str) -> WorkspaceObjectPermissions:
    w = WorkspaceClient()
    permissions = _call_workspace_api(
        w.workspace.get_permissions,
        {"workspace_object_type": "notebooks", "workspace_object_id": experiment_id},
    )
    return permissions


"""
Given a Permissions Object, and a list of users returns new permissions without the users
"""


def _remove_users_from_permissions_list(permissions, users):
    user_set = set(users)
    acls = permissions.access_control_list
    modified_acls = list(filter(lambda acl: acl.user_name not in user_set, acls))
    # No Changes as the user has no permissions on the endpoint
    if len(modified_acls) == len(acls):
        return None
    new_permissions = []
    for acl in modified_acls:
        for permission in acl.all_permissions:
            user = ()
            # "user_name", "group_name" and "service_principal_name" are all keywords used by the permission API later
            if acl.user_name is not None:
                user = ("user_name", acl.user_name)
            elif acl.group_name is not None:
                user = ("group_name", acl.group_name)
            else:
                user = ("service_principal_name", acl.service_principal_name)
            new_permissions.append((user, permission.permission_level))
    return new_permissions


# For a given a chain model name get all logged trace artifacts and return the corresponding experiment IDs
def _get_experiment_ids_from_trace_artifacts(model_name: str) -> List[str]:
    ml_artifacts = rest_get_review_artifacts(model_name)
    experiment_ids = _get_experiment_ids(
        _get_run_ids_from_artifact_uris(ml_artifacts.artifact_uris)
    )
    return experiment_ids


# Sets permissions on an endoint for the list of users
# Permissions is of type [((User_type, username), PermissionLevel)]
def _set_permissions_on_endpoint(
    endpoint_id: str,
    permissions: List[Tuple[Tuple[str, str], ServingEndpointPermissionLevel]],
):
    if permissions is None:
        return
    acls = []
    for users, permission_level in permissions:
        user_type, user = users
        acls.append(
            ServingEndpointAccessControlRequest.from_dict(
                {user_type: user, "permission_level": permission_level.value}
            )
        )
    # NOTE: THIS SHOULD ONLY BE CALLED ONCE
    # This endpoint performs a complete overwrite and should not be called more than once
    w = WorkspaceClient()
    _call_workspace_api(
        w.serving_endpoints.set_permissions,
        {
            "serving_endpoint_id": endpoint_id,
            "access_control_list": acls,
        },
    )


# Sets permission on experiment
# Permissions is of type [((User_type, username), PermissionLevel)]
def _set_permissions_on_experiment(
    experiment_id: str,
    permissions: List[Tuple[Tuple[str, str], ServingEndpointPermissionLevel]],
):
    if permissions is None:
        return
    acls = []
    for users, permission_level in permissions:
        user_type, user = users
        acls.append(
            WorkspaceObjectAccessControlRequest.from_dict(
                {user_type: user, "permission_level": permission_level.value}
            )
        )
    # NOTE: THIS SHOULD ONLY BE CALLED ONCE
    # This endpoint performs a complete overwrite and should not be called more than once
    w = WorkspaceClient()
    _call_workspace_api(
        w.workspace.set_permissions,
        {
            "workspace_object_type": "notebooks",
            "workspace_object_id": experiment_id,
            "access_control_list": acls,
        },
    )


# Update Permissions on Endpoint
def _update_permissions_on_endpoint(
    endpoint_id: str,
    users: List[str],
    permission_level: ServingEndpointPermissionLevel,
):
    w = WorkspaceClient()
    _call_workspace_api(
        w.serving_endpoints.update_permissions,
        {
            "serving_endpoint_id": endpoint_id,
            "access_control_list": [
                ServingEndpointAccessControlRequest(
                    user_name=user, permission_level=permission_level
                )
                for user in users
            ],
        },
    )


# Update Permissions on Experiment
def _update_permissions_on_experiment(
    experiment_ids: str,
    users: List[str],
    permission_level: Optional[WorkspaceObjectPermissionLevel] = None,
):
    w = WorkspaceClient()
    for experiment_id in experiment_ids:
        _call_workspace_api(
            w.workspace.update_permissions,
            {
                "workspace_object_type": "notebooks",
                "workspace_object_id": experiment_id,
                "access_control_list": [
                    WorkspaceObjectAccessControlRequest(
                        user_name=user,
                        permission_level=permission_level,
                    )
                    for user in users
                ],
            },
        )


def _get_endpoint_id_for_deployed_model(model_name: str):
    endpoint_ids = set()
    chain_deployments = get_deployments(model_name)
    w = WorkspaceClient()
    for deployment in chain_deployments:
        serving_endpoint = _call_workspace_api(
            w.serving_endpoints.get, {"name": deployment.endpoint_name}
        )
        endpoint_ids.add(serving_endpoint.id)
    return endpoint_ids


def _clear_permissions_for_user_endpoint(endpoint_id: str, clear_users: List[str]):
    # Retrieves all the permissions in the endpoint. Returned list is permission level mapping for all users
    permissions = _get_permissions_on_endpoint(endpoint_id)
    # Filter permissions list such that users in `clear_users` do not have any permissions.
    new_permissions = _remove_users_from_permissions_list(permissions, clear_users)
    # Re sets the permissions for the remaining users
    _set_permissions_on_endpoint(endpoint_id, new_permissions)


def _clear_permissions_for_user_experiments(
    experiment_ids: List[str], clear_users: List[str]
):
    for experiment_id in experiment_ids:
        # Retrieves all the permissions in the experiment. Returned list is permission level mapping for all users
        experiment_permissions = _get_permissions_on_experiment(experiment_id)
        # Filter permissions list such that users in `clear_users` do not have any permissions.
        new_permissions = _remove_users_from_permissions_list(
            experiment_permissions, clear_users
        )
        # Re sets the permisssions for the remaining users
        _set_permissions_on_experiment(experiment_id, new_permissions)


def set_permissions(
    model_name: str,
    users: List[str],
    permission_level: PermissionLevel,
):
    users_set = set(users)
    users = list(users_set)
    endpoint_ids = _get_endpoint_id_for_deployed_model(model_name)

    if len(endpoint_ids) == 0:
        raise ValueError("No deployments found for model_name " + model_name)
    # Set Permissions on Endpoints
    for endpoint_id in endpoint_ids:
        if permission_level == PermissionLevel.NO_PERMISSIONS:
            _clear_permissions_for_user_endpoint(endpoint_id, users)
        elif permission_level == PermissionLevel.CAN_VIEW:
            _update_permissions_on_endpoint(
                endpoint_id, users, ServingEndpointPermissionLevel.CAN_VIEW
            ),
        elif permission_level == PermissionLevel.CAN_QUERY:
            _update_permissions_on_endpoint(
                endpoint_id, users, ServingEndpointPermissionLevel.CAN_QUERY
            )
        elif permission_level == PermissionLevel.CAN_REVIEW:
            _update_permissions_on_endpoint(
                endpoint_id, users, ServingEndpointPermissionLevel.CAN_QUERY
            )
        elif permission_level == PermissionLevel.CAN_MANAGE:
            _update_permissions_on_endpoint(
                endpoint_id, users, ServingEndpointPermissionLevel.CAN_MANAGE
            )

    # Set permissions on Experiments if necessary
    experiment_ids = _get_experiment_ids_from_trace_artifacts(model_name)
    if permission_level == PermissionLevel.NO_PERMISSIONS:
        _clear_permissions_for_user_experiments(experiment_ids, users)
    elif permission_level == PermissionLevel.CAN_VIEW:
        # If the user previously had any permissions on the experiment delete them
        _clear_permissions_for_user_experiments(experiment_ids, users)
    elif permission_level == PermissionLevel.CAN_QUERY:
        # If the user previously had any permissions on the experiment delete them
        _clear_permissions_for_user_experiments(experiment_ids, users)
    elif permission_level == PermissionLevel.CAN_REVIEW:
        _update_permissions_on_experiment(
            experiment_ids, users, WorkspaceObjectPermissionLevel.CAN_READ
        )
    elif permission_level == PermissionLevel.CAN_MANAGE:
        # If the user previously had any permissions on the experiment delete them
        _update_permissions_on_experiment(
            experiment_ids, users, WorkspaceObjectPermissionLevel.CAN_READ
        )



# COMMAND ----------

import databricks
print(inspect.getsource(databricks.rag_studio.sdk))

# COMMAND ----------

dir(rag_studio)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## View deployments
# MAGIC
# MAGIC If you have lost the deployment information captured above, you can find it using `list_deployments()`.

# COMMAND ----------

# DBTITLE 1,View deployments
deployments = rag_studio.list_deployments()
for deployment in deployments:
  if deployment.model_name == uc_model_fqdn and deployment.model_version==uc_registered_chain_info.version:
    print(parse_deployment_info(deployment))
