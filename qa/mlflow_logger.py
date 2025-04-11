# qa/mlflow_logger.py

import mlflow
import time

def start_tracking(experiment_name="llm-qa-chatbot", run_name=None):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)

def log_basic_info(query, model, start_time, score=None, retrieved_docs=None, answer=None):
    duration = time.time() - start_time

    mlflow.log_param("query", query)
    mlflow.log_param("model", model)
    mlflow.log_metric("response_time", duration)

    if score is not None:
        mlflow.log_metric("confidence_score", score)

    if retrieved_docs:
        mlflow.log_param("retrieved_doc_count", len(retrieved_docs))
        mlflow.log_text("\n\n".join(retrieved_docs), "retrieved_docs.txt")

    if answer:
        mlflow.log_text(answer, "answer.txt")

def log_tags_and_status(confidence, threshold=0.3):
    if confidence < threshold:
        mlflow.set_tag("status", "low_confidence")
    else:
        mlflow.set_tag("status", "success")

def log_error(error_msg, query):
    mlflow.set_tag("status", "error")
    mlflow.set_param("query", query)
    mlflow.set_tag("error_type", error_msg)
