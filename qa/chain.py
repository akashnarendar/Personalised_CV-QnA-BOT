from qa.embedding import embed_query
from qa.retriever import retrieve_docs
from qa.llm import call_llm
from qa.mlflow_logger import start_tracking, log_basic_info, log_tags_and_status, log_error
import time


def get_answer_from_query(query):
    start = time.time()
    try:
        with start_tracking(run_name=query):
            query_vec = embed_query(query)
            docs = retrieve_docs(query_vec)

            if not docs:
                log_basic_info(query, "deepset/roberta-base-squad2", start_time=start)
                mlflow.set_tag("status", "no_context_found")
                return "❌ Sorry, I couldn't find any relevant information in the CV."

            answer, score = call_llm(query, docs)
            log_basic_info(query, "deepset/roberta-base-squad2", start, score, docs, answer)
            log_tags_and_status(score)

            return answer

    except Exception as e:
        log_error(str(e), query)
        return f"⚠️ Internal error: {str(e)}"
