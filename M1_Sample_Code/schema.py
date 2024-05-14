# Databricks notebook source
# MAGIC %md
# MAGIC # Current schema

# COMMAND ----------

from typing import Dict, List

# Currently, your chain's signature must follow this format.
def chain_predict_method(chat_messages: List[Message]) -> str:
    return "some response"

# OpenAI Chat Messages format
# https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
# Message type
chat_messages = {
        "messages": [ # The format for Message is a dict of {"role": "xx", "content": "xx"}
            {  # chat history
                "role": "user",
                "content": "What is RAG?",
            },
            {  # chat history
                "role": "assistant",
                "content": "Retrieval augmented generation is ...",
            },

            # .. repeated

            {  # user's current question, should always be last
                "role": "user",
                "content": "How do I try it?",
            },
        ]
    }

# The RAG Model Serving endpoint's API formats the returned str into OpenAI Chat Completions format
# https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-response
# https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format

api_return_value = {
  "id": "e9060529-7d6f-4a03-a8c8-bc19c3bbf78e", #generated unique ID
  "object": "chat.completion",
  "created": 1713853832, # timestamp
  "model": null, # not currently filled in
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "some response" # return value from your chain's predict method
      },
      "finish_reason": null # not currently filled in
    }
  ],
  "usage": { # not currently filled in
    "prompt_tokens": null,
    "completion_tokens": null,
    "total_tokens": null
  }
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Other options we are considering - please provide your feedback on these.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input signature
# MAGIC
# MAGIC These options are not mutually exclusive e.g., we could allow for more than one.

# COMMAND ----------

# Option 1: Allow the signature to include custom named parameters
def chain_predict_method(..., your_param_1: str, your_param_2: Dict[str, str], ...) -> str:
    return "some response"
  
# Option 2: Allow the signature to use `query_string` and `chat_history` as reserved names
# query_string is passed by the front-end app
# `chat_history` is the same format as `chat_messages`, but only contains the history e.g., messages earlier than the current `query_string`
def chain_predict_method(query_string, chat_history: List[Message]) -> str:
    return "some response"
  
# Option 3: Pass a debug=True parameter to get back a copy of the raw chain trace in the response
# The trace would be included in the endpoint's response
def chain_predict_method(..., return_traces=True, ...) -> str:
    return "some response"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output signature
# MAGIC
# MAGIC These options are not mutually exclusive e.g., we could allow for more than one.

# COMMAND ----------

# Option 1: Allow the user to return chat_completions directly - user is responsible for formatting the output

# Option 2A: Allow the return type to include custom parameters
def chain_predict_method(...) -> Tuple[str, List[str]]:
    return "some response", ["chunk_1_content", "chunk_2_content"]
# User would provide a mapping of the outputs to variable names & of which output is the response_string or chat_completions
mapping = {
  "response_string": 0,
  "custom_params": [1]
}
# Model Serving endpoint's API would format this into a dict.

# Option 2B: Allow the return type to be a Dict 
def chain_predict_method(...) -> Dict[str, str]:
    return {
    "query_response": "some response", 
    "retrieved_chunks": ["chunk_1_content", "chunk_2_content"]
    }
# User would provide a mapping of which output is the response_string or chat_completions
mapping = {
  "response_string": "query_response",
  "custom_params": ['retrieved_chunks']
}
# Model Serving endpoint's API would return this output as-is.
