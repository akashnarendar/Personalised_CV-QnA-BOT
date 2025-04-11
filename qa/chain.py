from qa.embedding import embed_query
from qa.retriever import retrieve_docs
from qa.llm import call_llm

def get_answer_from_query(query):
    query_vec = embed_query(query)
    docs = retrieve_docs(query_vec)

    if not docs:
        return "❌ Sorry, I couldn't find any relevant information in your document."

    return call_llm(query, docs)
