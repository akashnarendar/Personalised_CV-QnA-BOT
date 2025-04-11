from qa.embedding import embed_query
from qa.retriever import retrieve_docs
from qa.llm import call_llm

def get_answer_from_query(query):
    query_vec = embed_query(query)
    docs = retrieve_docs(query_vec)
    answer = call_llm(query, docs)
    return answer
