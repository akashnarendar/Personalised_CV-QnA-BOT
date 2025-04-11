import faiss
import numpy as np
import pickle
import os

_index = None
_doc_map = None

def load_index():
    global _index, _doc_map
    if _index is None:
        if not os.path.exists("data/index.faiss"):
            raise FileNotFoundError("FAISS index not found.")
        _index = faiss.read_index("data/index.faiss")
        with open("data/doc_map.pkl", "rb") as f:
            _doc_map = pickle.load(f)

def retrieve_docs(query_vec, top_k=3):
    load_index()
    D, I = _index.search(np.array([query_vec]), top_k)
    return [_doc_map[i] for i in I[0] if i != -1 and i in _doc_map]
