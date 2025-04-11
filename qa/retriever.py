# import faiss
# import numpy as np
# import pickle
# import os

# _index = None
# _doc_map = None

# def load_index():
#     global _index, _doc_map
#     if _index is None:
#         if not os.path.exists("data/index.faiss"):
#             raise FileNotFoundError("FAISS index not found.")
#         _index = faiss.read_index("data/index.faiss")
#         with open("data/doc_map.pkl", "rb") as f:
#             _doc_map = pickle.load(f)

# def retrieve_docs(query_vec, top_k=3):
#     load_index()
#     D, I = _index.search(np.array([query_vec]), top_k)
#     return [_doc_map[i] for i in I[0] if i != -1 and i in _doc_map]


import faiss
import numpy as np
import pickle
import os

# Paths to index and doc map
INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")
DOC_MAP_PATH = os.getenv("DOC_MAP_PATH", "data/doc_map.pkl")

_index = None
_doc_map = None

def load_index(force_reload=False):
    """
    Loads or reloads the FAISS index and doc map.

    Parameters:
    - force_reload: If True, reloads the index even if it's already loaded.
    """
    global _index, _doc_map

    if _index is not None and not force_reload:
        return

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"❌ FAISS index not found at: {INDEX_PATH}")
    
    if not os.path.exists(DOC_MAP_PATH):
        raise FileNotFoundError(f"❌ doc_map.pkl not found at: {DOC_MAP_PATH}")

    _index = faiss.read_index(INDEX_PATH)

    with open(DOC_MAP_PATH, "rb") as f:
        _doc_map = pickle.load(f)

    print("✅ FAISS index and document map loaded successfully.")

def retrieve_docs(query_vec, top_k=3):
    load_index()

    D, I = _index.search(np.array([query_vec]), top_k)
    return [_doc_map[i] for i in I[0] if i != -1 and i in _doc_map]
