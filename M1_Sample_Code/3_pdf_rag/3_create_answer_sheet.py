# Databricks notebook source
# MAGIC %md
# MAGIC ### Create an answer sheet from a RAG chain defined by a config YAML
# MAGIC
# MAGIC - Parameters:
# MAGIC     - An evaluation benchmark table name (assumes the benchmark table already exists in Unity Catalog) 
# MAGIC     - RAG chain notebook path
# MAGIC     - RAG chain config YAML path
# MAGIC     - An app version as arguments
# MAGIC
# MAGIC - Returns the answer sheet table name.
# MAGIC
# MAGIC **TODO**: Add an argument to specify the number of times to generate model responses.

# COMMAND ----------

# DBTITLE 1,Establish notebook parameters
dbutils.widgets.text("evaluation_benchmark_fqdn", "", label="Evaluation Benchmark UC FQDN")
dbutils.widgets.text("rag_chain_notebook_path",   "", label="RAG Chain Notebook Path")
dbutils.widgets.text("rag_chain_config_yaml",     "", label="RAG Chain Config YAML")
dbutils.widgets.text("app_version",               "", label="App Version")

# COMMAND ----------

# DBTITLE 1,Imports and configuration
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
import mlflow

import pandas as pd
from pyspark.sql.functions import pandas_udf

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog development;
# MAGIC use database rag_studio;

# COMMAND ----------

# DBTITLE 1,Run the small RAG chain driver notebook to create and log a new RAG chain based on the given configuration
rag_chain_notebook_path = dbutils.widgets.get("rag_chain_notebook_path")
rag_chain_config_yaml   = dbutils.widgets.get("rag_chain_config_yaml")

model_uri = dbutils.notebook.run("./3_rag_chain_driver_small_notebook", 600,
                                 arguments={"rag_chain_notebook_path":rag_chain_notebook_path,
                                            "rag_chain_config_yaml":rag_chain_config_yaml})
# @note: Run 3_rag_chain_driver_small_notebook to log the RAG chain and return the model URI.

# COMMAND ----------

# DBTITLE 1,Read evaluation benchmark table
evaluation_benchmark_fqdn  = dbutils.widgets.get("evaluation_benchmark_fqdn")
evaluation_benchmark_table = spark.table(evaluation_benchmark_fqdn)

# COMMAND ----------

display(evaluation_benchmark_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate a response for each question in the benchmark:

# COMMAND ----------

def query_RAG_chain_builder(model_uri):
    loaded_model = mlflow.langchain.load_model(model_uri)
    
    def query_RAG_chain(question):
        model_input = {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                },
                
            ]
        }

        # Run the model to see the output
        return loaded_model.invoke(model_input)

    return query_RAG_chain    
     
# @todo: Generate responses for each of the questions.

# COMMAND ----------

# model_uri       = 'runs:/3151ff0d68c342d091f576725765f911/chain'
# @todo: This needs to come from running the small driver notebook.

# qrc             = query_RAG_chain_builder(model_uri)
# query_RAG_chain = pandas_udf(qrc, StringType())

# COMMAND ----------

eval_test = evaluation_benchmark_table.toPandas()
qrc_test  = query_RAG_chain_builder(model_uri)

eval_test['response'] = eval_test.apply(lambda x: qrc_test(x['request']), axis=1, result_type='reduce')

# COMMAND ----------

# responses = evaluation_benchmark_table.\
#                 limit(1).\
#                 withColumn("response", query_RAG_chain("request")).\
#                 cache()
# # @todo: We can get into trouble with the rate of invocations to the RAG chain (which would call the DBRX serving endpoint in this case).

# display(responses)
# @todo: This seems to be erroring due to a serialization error - is this is because of the fact that the loaded model is in the query_RAG_chain function closure? 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write the responses to a table in UC

# COMMAND ----------

app_version = dbutils.widgets.get("app_version")

responses = spark.\
                createDataFrame(eval_test).\
                withColumn("app_version", F.lit(app_version)).\
                select('request_id', 'response', 'app_version')

# COMMAND ----------

answer_sheet_table_name = f"{evaluation_benchmark_fqdn}_responses_{app_version}"

responses.\
    write.\
    format('delta').\
    mode('overwrite').\
    saveAsTable(answer_sheet_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Return `answer_sheet_table_name` before exit

# COMMAND ----------

dbutils.notebook.exit(answer_sheet_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Static answer sheet (NOT USED)

# COMMAND ----------

# responses = [
#     {
#         'request_id':'sample_request_1',
#         'app_version': '1',
#         'response':"""The purpose of Southern California Edison Company's (SCE) rate case application is to request the establishment of new rates for the company. The application includes a proposed revenue requirement, a categorization of the proceeding, and a schedule for hearings and issues to be considered. The rate case application is a mechanism for SCE to recover its costs and earn a reasonable return on its investments, while also addressing safety and reliability concerns. The application is aligned with the Commission's modified Rate Case Plan schedule and includes testimony on SCE's corporate safety programs and relevant risk factors."""
#     },
#     {
#         'request_id':'sample_request_2',
#         'app_version': '1',
#         'response':"""SCE has stated several reasons for its request, including the need to recover costs associated with wildfire mitigation efforts, infrastructure replacement, and ensuring affordable electricity for all residents. The company has also proposed a framework for assessing the affordability of customer rates, which includes considering the benefits of proposed programs, selecting a reasonable and prudent scope and cost, implementing cost-control measures, and evaluating the impact on customers' ability to pay their electric bills."""
#     },
#     {
#         'request_id':'sample_request_3',
#         'app_version': '1',
#         'response':"""SCE's expenses are affected by regulations from both the CPUC and FERC. In its 2003 GRC, the CPUC adopted SCE's proposed method for splitting costs between those recovered through rates authorized by the CPUC and those authorized by FERC. This method has been followed in SCE's prior GRCs since 2006. Additionally, SCE's expenses include items subject to the ratemaking authority of the FERC."""
#     },
#     {
#         'request_id':'sample_request_4',
#         'app_version': '1',
#         'response':"""SCE will take several measures to mitigate wildfires. These include system hardening through the Wildfire Covered Conductor Program, improved situational awareness using high-definition cameras and weather stations, expanded inspections and vegetation management programs, and enhanced public outreach and operational practices. SCE will also maintain reasonable insurance coverage to protect customers from third-party wildfire claims and continue to purchase wildfire insurance to ensure that they have the funds necessary to operate the system and serve customers should further utility-related wildfire events occur."""
#     },
#     {
#         'request_id':'sample_request_5',
#         'app_version': '1',
#         'response':"""SCE assesses affordability when proposing rate increases, considering customer benefits and the impact on customers' ability to pay electric bills, with a special focus on income-qualified customers.
#         SCE has a four-part framework for assessing affordability, including evaluating the reasonableness of objectives, selecting a reasonable and prudent scope, pace, and cost, implementing cost-control measures, and considering the impact on customers' ability to pay.
#         SCE is committed to maintaining reasonable insurance coverage to protect customers from third-party wildfire claims and ensure the necessary funds to operate the system and serve customers.
#         SCE continues to invest in renewable energy resources and zero-carbon resources to supply 100% of all retail sales of electricity to California end-use customers.
#         SCE is dedicated to performing crucial functions in a manner that is affordable for customers and is fundamentally committed to spending customer dollars wisely and prudently to provide commensurate value for the important services that they provide."""
#     },
#     {
#         'request_id':'sample_request_6',
#         'app_version': '1',
#         'response':"""Electric utilities generate electricity through various methods, including burning fossil fuels such as coal or natural gas, harnessing the power of water through hydroelectric dams, using nuclear reactions, or utilizing renewable resources like wind and solar power. The generated electricity is then transmitted and distributed to customers through a network of power lines and substations."""
#     },
#     {
#         'request_id':'sample_request_7',
#         'app_version': '1',
#         'response':"""The major fuel sources for electricity generation, as mentioned in the context, include fossil fuels, hydro, and solar. The context also mentions Palo Verde, which is a nuclear power plant. However, it's not clear if nuclear power is considered a major fuel source in this context."""
#     },
#     {
#         'request_id':'sample_request_8',
#         'app_version': '1',
#         'response':"""SCE operates 33 hydroelectric plants, one combined-cycle gas plant with two units, five gas-fired peaker units, a diesel-driven electric generating plant, and 25 solar photovoltaic sites. Additionally, SCE has a 78.21% interest in San Onofre Nuclear Generating Station (SONGS) Units 2 and 3, and a 15.8% interest in Palo Verde Nuclear Generating Station Units 1, 2, and 3."""
#     },
#     {
#         'request_id':'sample_request_9',
#         'app_version': '1',
#         'response':"""SCE maintains over 4,000 miles of transmission lines."""
#     },
#     {
#         'request_id':'sample_request_10',
#         'app_version': '1',
#         'response':"""I don't have real-time data access, but based on the provided context, I can't determine the exact population of the area served by Southern California Edison Company (SCE). The context provided is about SCE's earnings, rates, and property, not the population of the service area."""
#     }
# ]

# COMMAND ----------

############
# Turn the eval dataset into a Delta Table
############
# TODO: Change these values to your catalog and schema
# uc_catalog          = "development"
# uc_schema           = "rag_studio"
# eval_table_name     = "rate_case_eval_set"
# response_table_name = 'rate_case_responses'

# response_table_fqdn = f"{uc_catalog}.{uc_schema}.{response_table_name}"

# df2 = spark.read.json(spark.sparkContext.parallelize(responses))
# df2.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
#     response_table_fqdn
# )
# print(f"Eval set written to: {response_table_fqdn}")

# COMMAND ----------

# dbutils.notebook.exit(response_table_fqdn)
