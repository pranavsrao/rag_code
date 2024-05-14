# Databricks notebook source
# MAGIC %md
# MAGIC ### Create an evaluation benchmark with the following columns `request_id`, `request`, `expected_response` based on the SCE spreadsheet provided to us. 

# COMMAND ----------

# DBTITLE 1,Establish notebook parameters
dbutils.widgets.text("evaluation_benchmark_table_name", "", label="Evaluation Benchmark Table Name")
# @note: Parameterized this notebook to accept a name for the new benchmark table.

# COMMAND ----------

# DBTITLE 1,Imports and configuration
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog development;
# MAGIC use database rag_studio;

# COMMAND ----------

# DBTITLE 1,Read information from SCE spreadsheet provided to us
grc_benchmark_table = "bronze_grc_benchmark" 
grc_benchmark       = spark.table(f"development.rag_studio.{grc_benchmark_table}")

# COMMAND ----------

# display(grc_benchmark)

# COMMAND ----------

# DBTITLE 1,Write benchmark table
benchmark_table = grc_benchmark.select("request", 
                             "expected_response").\
                                distinct().\
                                withColumn("request_id", F.monotonically_increasing_id()).\
                                select("request_id", "request", "expected_response")

# COMMAND ----------

evaluation_benchmark_table_name = dbutils.widgets.get("evaluation_benchmark_table_name")

benchmark_table.write.\
    format('delta').\
    mode('overwrite').\
    saveAsTable(evaluation_benchmark_table_name)

# COMMAND ----------

# DBTITLE 1,Return the UC FQDN of the evaluation benchmark table
dbutils.notebook.exit(f"development.rag_studio.{evaluation_benchmark_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Previous evaluation benchmark (NOT USED)

# COMMAND ----------

# eval_dataset = [
#     {
#         "request_id": "sample_request_1",
#         "request": "What is the purpose of SCE's rate case application?",
#         # Expected retrieval context is optional, if not provided, RAG Studio will use LLM judge to assess each retrieved context
#         # Expected response is optional
#         "expected_response": """The purpose of Southern California Edison Company’s (SCE) rate case application is to request an authorized base revenue requirement (ABRR) of $7.601 billion for the test year 2021, which would reflect an increase in rates to support various objectives. These include maintaining and improving the electric grid, implementing the state’s public policy goals such as wildfire mitigation and de-carbonization through electrification, and adapting to a rapidly modernizing grid. The application also seeks to recover expenditures recorded in various wildfire-related memorandum accounts for 2019 and 20203. Additionally, SCE aims to continue providing safe, reliable, affordable, and increasingly clean electricity to its customers while addressing the challenges posed by the global climate crisis and catastrophic wildfires."""
#     },
#     {
#         "request_id": "sample_request_2",
#         "request": "What are different reasons SCE has stated for its request?",
#         "expected_response": """The document outlines several reasons for Southern California Edison’s (SCE) request in the General Rate Case Application: Safety and Reliability: SCE emphasizes the need to maintain and improve the grid to ensure safety and reliability for customers, especially in light of the increased risk of catastrophic wildfires.Wildfire Mitigation: The company proposes comprehensive programs and actions aimed at mitigating wildfire risks associated with their equipment due to extreme environmental conditions.Regulatory Compliance: SCE’s request includes necessary expenditures to comply with new regulations and legislative requirements, such as those from Assembly Bill 1054. Clean Energy Goals: SCE is committed to supporting the state’s clean energy policies, including the reduction of greenhouse gas emissions and the integration of distributed energy resources."""
#     },
#     {
#         "request_id": "sample_request_3",
#         "request":  "Which regulations affect SCE's expenses?",
#         "expected_response": """The document outlines several regulations that impact Southern California Edison Company’s (SCE) expenses: Fire-Threat Safety Map OIR: Regulations for electric power lines in high fire-threat areas, including new inspection, maintenance, and vegetation management regulations. Risk Assessment Mitigation Phase (RAMP): SCE’s RAMP Report examines top safety risks and identifies new mitigations, influencing operational and capital expenditures. Grid Safety & Resiliency Program (GSRP): SCE’s GSRP Application proposes wildfire mitigation programs and activities, impacting costs related to grid hardening. Wildfire Mitigation Plan (WMP): SCE’s WMP describes programs and activities for wildfire risk mitigation, affecting wildfire-related costs. These regulations require SCE to undertake specific actions and programs that contribute to their overall expenses."""
#     },
#     {
#         "request_id": "sample_request_4",
#         "request": "Which measures will SCE take to mitigate wildfires?",
#         "expected_response": """The document outlines Southern California Edison Company’s (SCE) measures to mitigate wildfires and the associated costs as follows. System Hardening: SCE’s Grid Safety & Resiliency Program (GSRP) and Wildfire Mitigation Program (WMP) include grid resiliency measures like the Wildfire Covered Conductor Program (WCCP) to replace and insulate wires. Situational Awareness: Implementation of high-definition cameras and weather stations to detect and monitor ignitions. Vegetation Management: Expanded programs to increase pruning clearances and remove hazardous trees. Public Outreach and PSPS: Educating customers about wildfire threats and selectively de-energizing lines through the Public Safety Power Shutoff (PSPS) program."""
#     },
#     {
#         "request_id": "sample_request_5",
#         "request": "What measures is SCE taking to protect low-income customers from increased electricity costs?",
#         "expected_response": """The document outlines SCE’s commitment to affordability, especially for low-income customers, in the context of their General Rate Case (GRC) application. Here are the key points: Affordability Assessment: SCE evaluates the impact of rate increases on all customers, with a particular focus on low-income customers.Four-Part Framework: SCE proposes a framework to assess affordability, which includes evaluating the reasonableness of objectives, scope, pace, and cost, as well as the impact on customers’ ability to pay their bills. Income-Qualified Customers: Special attention is given to income-qualified customers to ensure they are not overburdened by energy expenditures. Policy Compliance: SCE’s efforts align with the state policy that essential electricity should be affordable for all residents. SCE’s approach is to balance the need for infrastructure investments with the importance of keeping electricity affordable for low-income customers."""
#     },
#     {
#         'request_id': 'sample_request_6',
#         'request': "How does an electric utility generate electricity?",
#         'expected_response': """Though SCE is an electric utility, the question is irrelevant as it addresses all electric utilities and does not directly pertain to why SCE is proposing changes to its electricity rates."""
#     },
#     {
#         'request_id':'sample_request_7',
#         'request':"What are major fuel sources for electricity generation?",
#         'expected_response':"""This question pertains to the general topic of electricity generation and is directly related to why SCE is proposing changes to its electricity rates."""
#     },    
#     {
#         'request_id':'sample_request_8',
#         'request':"How many generating stations does SCE operate?",
#         'expected_response':"""This question is not directly related to why SCE is proposing changes to its electricity rates."""
#     },
#     {
#         'request_id':'sample_request_9',
#         'request':"How many miles of transmission lines does SCE maintain?",
#         'expected_response': """This question is not directly related to why SCE is proposing changes to its electricity rates."""
#     },
#     {
#         'request_id':'sample_request_10',
#         'request':"What is the population of the area which SCE services?",
#         'expected_response': """This question is not directly related to why SCE is proposing changes to its electricity rates."""
#     },
# ]

# COMMAND ----------

# ############
# # Turn the eval dataset into a Delta Table
# ############
# # TODO: Change these values to your catalog and schema
# uc_catalog          = "development"
# uc_schema           = "rag_studio"
# eval_table_name     = "rate_case_eval_set"
# response_table_name = 'rate_case_responses'
# eval_table_fqdn = f"{uc_catalog}.{uc_schema}.{eval_table_name}"

# df = spark.read.json(spark.sparkContext.parallelize(eval_dataset))
# df.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
#     eval_table_fqdn
# )
# print(f"Eval set written to: {eval_table_fqdn}")

# COMMAND ----------

# dbutils.exit(eval_table_fqdn)
