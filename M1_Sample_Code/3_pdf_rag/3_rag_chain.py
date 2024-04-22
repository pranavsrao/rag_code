# Databricks notebook source
# DBTITLE 1,Databricks Rag Studio Installer
# MAGIC %run ../wheel_installer

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

# Before logging this chain using the driver notebook, you need to comment out this line.
# dbutils.library.restartPython() 

# COMMAND ----------

# DBTITLE 1,Import packages
from operator import itemgetter

from databricks import rag
from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# COMMAND ----------

# DBTITLE 1,Multistage Chat Joke Generator
############
# RAG Studio requires your chain to accept an array of OpenAI-formatted messages as a `messages` parameter. Schema: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage
# These helper functions help parse the `messages` array
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


############
# Get the configuration YAML
############
rag_config = rag.RagConfig("3_rag_chain_config.yaml")


############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=rag_config.get("vector_search_endpoint_name"),
    index_name=rag_config.get("vector_search_index"),
)
vector_search_schema = rag_config.get("vector_search_schema")

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=vector_search_schema.get("chunk_text"),
    columns=[
        vector_search_schema.get("primary_key"),
        vector_search_schema.get("chunk_text"),
        vector_search_schema.get("document_source"),
    ],
).as_retriever(search_kwargs=rag_config.get("vector_search_parameters"))

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############
rag.set_vector_search_schema(
    primary_key=vector_search_schema.get("primary_key"),
    text_column=vector_search_schema.get("chunk_text"),
    doc_uri=vector_search_schema.get(
        "document_source"
    ),  # Review App uses `doc_uri` to display chunks from the same document in a single view
)

############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = rag_config.get("chunk_template")
    chunk_contents = [chunk_template.format(chunk_text=d.page_content) for d in docs]
    return "".join(chunk_contents)


############
# Prompt Template for generation
############
prompt = PromptTemplate(
    template=rag_config.get("chat_prompt_template"),
    input_variables=rag_config.get("chat_prompt_template_variables"),
)

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=rag_config.get("chat_endpoint"),
    extra_params=rag_config.get("chat_endpoint_parameters"),
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

############
# Test the Chain
############
# model_input_sample = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "What is the purpose of SCE's rate case application?",
#         }
#     ]
# }
# print(chain.invoke(model_input_sample))

############
# Tell RAG Studio about the chain - required for logging, but not local testing of this chain
############
rag.set_chain(chain)

# COMMAND ----------

help(StrOutputParser)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experimental: view trace locally
# MAGIC
# MAGIC ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
# MAGIC
# MAGIC We are actively improving the trace logging schema - the schema will change in the next release.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **IMPORTANT: This code must be commented out in order to log the model**

# COMMAND ----------

# DBTITLE 1,RAG Experimental Execution Command
# %run ../RAG_Experimental_Code

# COMMAND ----------

# DBTITLE 1,Escaped JSON Display Generator

# import json
# import html

# json_trace = experimental_get_json_trace(chain, model_input_sample)

# json_string = json.dumps(json_trace, indent=4)

# # Escape HTML characters to avoid XSS or rendering issues
# escaped_json_string = html.escape(json_string)

# # Generate HTML with the escaped JSON inside <pre> and <code> tags
# pretty_json_html = f"<html><body><pre><code>{escaped_json_string}</code></pre></body></html>"

# # To use the HTML string in a context that renders HTML, 
# # such as a web application or a notebook cell that supports HTML output
# displayHTML(pretty_json_html)
