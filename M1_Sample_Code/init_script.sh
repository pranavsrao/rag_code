pip install $RAG_STUDIO_WHL
pip install $RAG_EVAL_WHL
pip install opentelemetry-api opentelemetry-sdk langchain -U -q
pip uninstall mlflow mlflow-skinny -y # uninstall existing mlflow to avoid installation issues
pip install $MLFLOW_WHL -U
pip install $MLFLOW_SKINNY_WHL -U