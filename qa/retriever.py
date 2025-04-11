import faiss
import numpy as np
import pickle

# Load FAISS index and doc map
index = faiss.read_index("data/index.faiss")
with open("data/doc_map.pkl", "rb") as f:
    doc_map = pickle.load(f)

def retrieve_docs(query_vec, top_k=3):
    D, I = index.search(np.array([query_vec]), top_k)
    return [doc_map[i] for i in I[0]]