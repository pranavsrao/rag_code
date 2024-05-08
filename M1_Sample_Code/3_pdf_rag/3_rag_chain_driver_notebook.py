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

# MAGIC %pip install spacy
# MAGIC %pip install textstat

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
chain_config_file   = "3_rag_chain_config.yaml"
chain_notebook_file = "3_rag_chain"

chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)
chain_config_path   = os.path.join(os.getcwd(), chain_config_file)

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
mlflow.langchain.autolog()
############
# Test the model locally
# This is the same input that the REST API will accept once deployed.
############

model_input = {
    "messages": [
        {
            "role": "user",
            "content": "Where does SCE operate?",
        },
        
    ]
}

loaded_model = mlflow.langchain.load_model(logged_chain_info.model_uri)

# Run the model to see the output
loaded_model.invoke(model_input)

# COMMAND ----------

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

escaped_json_string = html.escape(json.dumps(ground_truth_example, indent=4))

# Generate HTML with the escaped JSON inside <pre> and <code> tags
pretty_json_html = f"<html><body><pre><code>{escaped_json_string}</code></pre></body></html>"

# To use the HTML string in a context that renders HTML, 
# such as a web application or a notebook cell that supports HTML output
displayHTML(pretty_json_html)

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

# COMMAND ----------

############
# Turn the eval dataset into a Delta Table
############
# TODO: Change these values to your catalog and schema
uc_catalog      = "development"
uc_schema       = "rag_studio"
eval_table_name = "rate_case_eval_set"
eval_table_fqdn = f"{uc_catalog}.{uc_schema}.{eval_table_name}"

# df = spark.read.json(spark.sparkContext.parallelize(eval_dataset))
# df.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
#     eval_table_fqdn
# )
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
uc_catalog    = "development"
uc_schema     = "rag_studio"
model_name    = "rate_case_2"
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

# COMMAND ----------

rag_studio.list_deployments()

# COMMAND ----------

# MAGIC %md ### Inspect payload table:

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog development;
# MAGIC use database rag_studio;

# COMMAND ----------

payload_table = spark.table('`rag_studio-rate_case_2_payload`')

# COMMAND ----------

display(payload_table.orderBy(F.desc('timestamp_ms')))
