# qa/chain.py

from qa.embedding import embed_query
from qa.retriever import retrieve_docs
from qa.llm import call_llm
import time
import mlflow

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("llm-qa-chatbot")

def get_answer_from_query(query):
    with mlflow.start_run():
        start = time.time()
        query_vec = embed_query(query)
        docs = retrieve_docs(query_vec)

        if not docs:
            mlflow.log_param("query", query)
            mlflow.log_param("model", "distilbert-base-uncased-distilled-squad")
            mlflow.log_metric("response_time", time.time() - start)
            mlflow.log_text("No documents matched the query.", "retrieved_docs.txt")
            return "❌ Sorry, I couldn't find any relevant information in your document."

        answer = call_llm(query, docs)
        duration = time.time() - start

        # 🔍 Log all details to MLflow
        mlflow.log_param("query", query)
        mlflow.log_param("model", "distilbert-base-uncased-distilled-squad")
        mlflow.log_param("retrieved_doc_count", len(docs))
        mlflow.log_metric("response_time", duration)
        mlflow.log_text("\n\n".join(docs), "retrieved_docs.txt")
        mlflow.log_text(answer, "answer.txt")

        return answer
