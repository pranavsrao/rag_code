# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook for creating an evaluation benchmark for rate case Q&A
# MAGIC
# MAGIC ### The following process was used to create the evaluation benchmark:
# MAGIC 1. Created 5 relevant seed questions, 5 irrelevant seed questions 
# MAGIC 2. Generated ground truth answers to relevant seed questions using Copilot. However, answers from Copilot were manually cross-checked by referencing the document.
# MAGIC 3. Add context (added as list of strings). Only 
# MAGIC 4. Add responses from RAG application (see llamaindex-RAG-prototype notebook)
# MAGIC 5. Add expected score
# MAGIC
# MAGIC **Should different responses be simulated with varying scores?**

# COMMAND ----------

!pip install wget
!pip install llama-index
!pip install llama-index-readers-file pymupdf
!pip install llama-index-vector-stores-azureaisearch
!pip install azure-search-documents==11.4.0 llama-index-embeddings-azure-openai llama-index-llms-azure-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Questions

# COMMAND ----------

# Relevant questions:
relevant_qs = ["What is the purpose of SCE's rate case application?",
               "What are different reasons SCE has stated for its request?",
               "Which regulations affect SCE's expenses?",
               "Which measures will SCE take to mitigate wildfires?",
               "What measures is SCE taking to protect low-income customers from increased electricity costs?"]

# Irrelevant questions:
irrelevant_qs = ["How does an electric utility generate electricity?",
                 "What are major fuel sources for electricity generation?",
                 "How many generating stations does SCE operate?",
                 "How many miles of transmission lines does SCE maintain?",
                 "What is the population of the area which SCE services?"]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ground Truths

# COMMAND ----------

ground_truths = ["""The purpose of Southern California Edison Company’s (SCE) rate case application is to request an authorized base revenue requirement (ABRR) of $7.601 billion for the test year 2021, which would reflect an increase in rates to support various objectives. These include maintaining and improving the electric grid, implementing the state’s public policy goals such as wildfire mitigation and de-carbonization through electrification, and adapting to a rapidly modernizing grid. The application also seeks to recover expenditures recorded in various wildfire-related memorandum accounts for 2019 and 20203. Additionally, SCE aims to continue providing safe, reliable, affordable, and increasingly clean electricity to its customers while addressing the challenges posed by the global climate crisis and catastrophic wildfires.""", 
"""The document outlines several reasons for Southern California Edison’s (SCE) request in the General Rate Case Application: Safety and Reliability: SCE emphasizes the need to maintain and improve the grid to ensure safety and reliability for customers, especially in light of the increased risk of catastrophic wildfires.Wildfire Mitigation: The company proposes comprehensive programs and actions aimed at mitigating wildfire risks associated with their equipment due to extreme environmental conditions.Regulatory Compliance: SCE’s request includes necessary expenditures to comply with new regulations and legislative requirements, such as those from Assembly Bill 1054. Clean Energy Goals: SCE is committed to supporting the state’s clean energy policies, including the reduction of greenhouse gas emissions and the integration of distributed energy resources.""", """The document outlines several regulations that impact Southern California Edison Company’s (SCE) expenses: Fire-Threat Safety Map OIR: Regulations for electric power lines in high fire-threat areas, including new inspection, maintenance, and vegetation management regulations. Risk Assessment Mitigation Phase (RAMP): SCE’s RAMP Report examines top safety risks and identifies new mitigations, influencing operational and capital expenditures. Grid Safety & Resiliency Program (GSRP): SCE’s GSRP Application proposes wildfire mitigation programs and activities, impacting costs related to grid hardening. Wildfire Mitigation Plan (WMP): SCE’s WMP describes programs and activities for wildfire risk mitigation, affecting wildfire-related costs. These regulations require SCE to undertake specific actions and programs that contribute to their overall expenses.""",
"""The document outlines Southern California Edison Company’s (SCE) measures to mitigate wildfires and the associated costs as follows. System Hardening: SCE’s Grid Safety & Resiliency Program (GSRP) and Wildfire Mitigation Program (WMP) include grid resiliency measures like the Wildfire Covered Conductor Program (WCCP) to replace and insulate wires. Situational Awareness: Implementation of high-definition cameras and weather stations to detect and monitor ignitions. Vegetation Management: Expanded programs to increase pruning clearances and remove hazardous trees. Public Outreach and PSPS: Educating customers about wildfire threats and selectively de-energizing lines through the Public Safety Power Shutoff (PSPS) program.""",
"""The document outlines SCE’s commitment to affordability, especially for low-income customers, in the context of their General Rate Case (GRC) application. Here are the key points: Affordability Assessment: SCE evaluates the impact of rate increases on all customers, with a particular focus on low-income customers.Four-Part Framework: SCE proposes a framework to assess affordability, which includes evaluating the reasonableness of objectives, scope, pace, and cost, as well as the impact on customers’ ability to pay their bills. Income-Qualified Customers: Special attention is given to income-qualified customers to ensure they are not overburdened by energy expenditures. Policy Compliance: SCE’s efforts align with the state policy that essential electricity should be affordable for all residents. SCE’s approach is to balance the need for infrastructure investments with the importance of keeping electricity affordable for low-income customers.""", """Though SCE is an electric utility, the question is irrelevant as it addresses all electric utilities and does not directly pertain to why SCE is proposing changes to its electricity rates.""",
"""This question pertains to the general topic of electricity generation and is directly related to why SCE is proposing changes to its electricity rates.""",
"""This question is not directly related to why SCE is proposing changes to its electricity rates.""",
"""This question is not directly related to why SCE is proposing changes to its electricity rates.""",
"""This question is not directly related to why SCE is proposing changes to its electricity rates."""]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Context

# COMMAND ----------

context = [
    [
        """In this application, SCE asks the California Public Utilities Commission (CPUC or Commission) for an authorized base revenue requirement (ABRR) of $7.601 billion to become effective January 1, 2021, and to reflect the ABRR in distribution, generation and new system generation rates.""",
        """ As discussed in Section II.B below, SCE also requests authorization through separate sub-tracks of this proceeding to recover 2019 and 2020 recorded expenditures currently being tracked in various wildfire-related memorandum accounts.""",
        """That mission is not changing in this GRC cycle, and as the effects of the global climate crisis are being experienced in our region, SCE has provided and continues to provide the infrastructure and programs necessary to implement the State’s ambitious public policy goals, including wildfire mitigation, de-carbonization of the economy through electrification, and integration of distributed energy resources across a rapidly modernizing grid. """
    ],
    [
        """The bulk of SCE’s revenue requirement request in this GRC relates to the foundational work that SCE has always performed to maintain and improve the grid and the support functions necessary to provide our services, while continuing the investments necessary to implement the State’s primary policy objective to reduce greenhouse gas (GHG) emissions. SCE is dedicated to performing these crucial functions in a manner that is affordable for customers and is fundamentally committed to spending customer dollars wisely and prudently to provide commensurate value for the important services that we provide. As the Legislature recently recognized, SCE needs to continue to do the important work and make the investments that are crucial to fostering the health of the California economy. SCE’s requests in this GRC are vital to that goal.""",
        """SCE proposes the continuation of comprehensive programs, activities, and actions aimed at significantly mitigating and minimizing the risk of wildfire associated with our equipment in light of more extreme environmental conditions and other factors.""",
        """Consistent with the Legislature’s recent direction in AB 1054, SCE is also continuing to purchase wildfire insurance to protect customers from third-party wildfire claims, and to ensure that we have the funds necessary to operate the system and serve customers should further utility-related wildfire events occur notwithstanding our best efforts to prevent them.""",
        """ SCE must continue to maintain its largely carbon-free utility-owned generation fleet and adapt our grid and operational processes to, for example, integrate Distributed Energy Resources (DERs), increase electrification, and support customer retail choice in order to advance California’s clean energy policies aimed at combatting the climate crisis, and to support the evolving energy economy."""
    ],
    [
        """In late 2017 the Commission concluded an OIR that adopted new regulations to enhance the fire safety of electric power lines located in high fire-threat areas.10 In D.17-12-024, the Commission adopted new inspection, maintenance and vegetation management regulations; created what is known as the High Fire Threat District (HFTD) statewide fire map; and authorized the utilities to track the costs they incur to implement the regulations and seek cost recovery in future applications. """,
        """SCE submitted its Risk Assessment Mitigation Phase (RAMP) Report in November 2018. That RAMP Report examined the top safety risks to our customers and the communities we are privileged to serve, to our company, and to our employees and contractors.""",
        """On September 10, 2018, prior to the passage of SB 901, SCE filed A.18-09-002, an application requesting approval of proposed wildfire mitigation programs and activities (and associated costs) related to “grid hardening” over the 2018-2020 time horizon. The filing, known as the Grid Safety & Resiliency Program (GSRP) Application, was meant to serve as a “bridge” between the 2018 and 2021 GRCs for critical wildfire mitigation work necessary to protect public safety. """,
        """On February 6, 2019, SCE filed its 2019 Wildfire Mitigation Plan (WMP) in R.18-10-007, describing the programs and activities it intended to pursue in 2019 related to wildfire risk mitigation."""
    ],
    [
        """System hardening – SCE’s Grid Safety & Resiliency Program (GSRP), Wildfire Mitigation Program (WMP), and efforts in this rate case all include robust grid resiliency measures, centered around the implementation of our Wildfire Covered Conductor Program (WCCP). """,
        """Improved situational awareness – SCE is pursuing additional situational awareness programs, including the use of high-definition cameras and weather stations to detect and monitor any ignitions that do occur to prevent the spread of fire and limit its consequences. Expanded inspections and vegetation management programs – SCE has implemented expanded infrastructure inspection programs.""",
        """ It is critical for SCE to continue to educate customers about the threat of wildfire and to closely coordinate with local fire agencies and communities during periods of extreme fire danger such as Red Flag Warning days. During those and other appropriate risk-informed times, and in order to protect customers, SCE will also proactively and selectively de-energize lines through its new Public Safety Power Shutoff (PSPS) program."""
    ],
    [
        """when SCE puts forward any request to increase customer rates, including in this GRC, one of the key factors we assess is affordability and the associated impact of our requests on customer rates and bills. In assessing affordability in this proceeding, SCE urges the Commission to consider the following proposed four-part framework: 1) Are the objectives that SCE proposes reasonable when considering the customer benefits the programs provide? 2) Has SCE selected a reasonable and prudent scope, pace and cost to achieve these objectives? 3) Has SCE undertaken cost-control measures to reduce, to the extent practicable, the cost impact on customers overall? 4) How does the request impact SCE customers’ ability to pay their electric bills (with a special focus on income-qualified customers)?"""
    ]
]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Responses:

# COMMAND ----------

import logging
import sys
from pathlib import Path

from IPython.display import Markdown

from azure.core.credentials         import AzureKeyCredential
from azure.search.documents         import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import (
    # SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.settings import Settings

from llama_index.llms.azure_openai           import AzureOpenAI
from llama_index.embeddings.azure_openai     import AzureOpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.azureaisearch import (
    IndexManagement,
    MetadataIndexFieldType,
)

# COMMAND ----------

aoai_endpoint    = "https://azure-openai-llmops-framework.openai.azure.com/"
aoai_api_key     = f"{dbutils.secrets.get('kv-llmops-framework', 'aoai-api-key')}"
aoai_api_version = "2023-09-01-preview"

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-turbo",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
)
# @note: Completion model.

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
)
# @note: Embedding model.

# COMMAND ----------

search_service_endpoint    = "https://search-llmops-framework.search.windows.net"
search_service_api_key     = f"{dbutils.secrets.get('kv-llmops-framework', 'search-service-api-key')}"
search_service_api_version = "2023-11-01"
credential                 = AzureKeyCredential(search_service_api_key)

# Index name to use
index_name = "rate-case-rag"

# Use index client to demonstrate creating an index:
index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=credential,
)

# Use search client to demonstrate using existing index:
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=credential,
)

# COMMAND ----------

vector_store = AzureAISearchVectorStore(
    search_or_index_client         = index_client,
    index_name                     = index_name,
    index_management               = IndexManagement.CREATE_IF_NOT_EXISTS,
    id_field_key                   = "id",
    chunk_field_key                = "chunk",
    embedding_field_key            = "embedding",
    embedding_dimensionality       = 1536,
    metadata_string_field_key      = "metadata",
    doc_id_field_key               = "doc_id",
    language_analyzer              = "en.lucene",
    vector_algorithm_type          = "exhaustiveKnn",
)

# COMMAND ----------

from llama_index.core            import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing                      import Any, List

from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing                  import Optional

class VectorDBRetriever(BaseRetriever):
    """Retriever over a Azure AI Search vector store."""

    def __init__(
        self,
        vector_store:     AzureAISearchVectorStore,
        embed_model:      Any,
        query_mode:       str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store     = vector_store
        self._embed_model      = embed_model
        self._query_mode       = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


# COMMAND ----------

retriever = VectorDBRetriever(vector_store, embed_model, query_mode="default", similarity_top_k=2)

# COMMAND ----------

from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

# COMMAND ----------

# Responses:
import pandas as pd
eval_df = pd.DataFrame({'question': relevant_qs + irrelevant_qs,
                        'ground_truth': ground_truths,
                        'context': context + [[""]]*5})

# COMMAND ----------

eval_df['response'] = eval_df['question'].apply(lambda x: query_engine.query(x))

# COMMAND ----------

# Responses are currently stored as response type objects containing metadata
# Need to convert to string to pull only GPT response
eval_df['response'] = eval_df['response'].astype(str)

# COMMAND ----------

def display_markdown(obj):
    display(Markdown(f"<b>{obj}</b>"))

# COMMAND ----------

for i in range(len(eval_df)):
    display_markdown(eval_df.iloc[i]['question'])
    display_markdown(eval_df.iloc[i]['response'])
    print()

# COMMAND ----------

# MAGIC %md #### Expected scores:

# COMMAND ----------

for i in range(len(eval_df)):
    display_markdown(eval_df.iloc[i]['question'])
    display_markdown(eval_df.iloc[i]['ground_truth'])
    display_markdown(eval_df.iloc[i]['response'])
    print()

# COMMAND ----------

expected_scores = [2, 1, 1, 4, 4, 1, 1, 1, 1, 1]

# COMMAND ----------

eval_df['expected_score'] = pd.Series(expected_scores)

# COMMAND ----------

eval_df

# COMMAND ----------

sp_eval_df = spark.createDataFrame(eval_df)

sp_eval_df

# COMMAND ----------

sp_eval_df.write.mode("overwrite").format("delta").saveAsTable(
'dev.llmops_framework.rate_case_eval'
)

# COMMAND ----------


