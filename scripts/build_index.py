# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import os, pickle

# model = SentenceTransformer('all-MiniLM-L6-v2')

# cv_path = "data/docs/Narendar_Punithan_CV_final_word.txt"
# with open(cv_path, "r") as f:
#     raw_text = f.read()

# # Split by double newlines and clean each chunk
# raw_chunks = raw_text.split("\n\n")
# doc_chunks = [chunk.strip().replace("\n", " ") for chunk in raw_chunks if len(chunk.strip()) > 50]

# print(f"[INFO] Loaded {len(doc_chunks)} meaningful CV chunks.")

# # Create vector embeddings
# embeddings = model.encode(doc_chunks)

# # Initialize FAISS
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(np.array(embeddings))

# # Save index + mapping
# faiss.write_index(index, "data/index.faiss")
# with open("data/doc_map.pkl", "wb") as f:
#     pickle.dump({i: chunk for i, chunk in enumerate(doc_chunks)}, f)

# print("[INFO] FAISS index and doc map saved successfully.")


import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Define the correct path to your CV file (mounted from ./data/docs/)
cv_path = os.getenv(
    "CV_PATH",
    "/opt/airflow/data/docs/Narendar_Punithan_CV_final_word.txt"
)

# ✅ Where to save the index and doc map (shared volume)
index_path = os.getenv(
    "INDEX_PATH",
    "/opt/airflow/data/index.faiss"
)
doc_map_path = os.getenv(
    "DOC_MAP_PATH",
    "/opt/airflow/data/doc_map.pkl"
)

# Load CV
if not os.path.exists(cv_path):
    raise FileNotFoundError(f"❌ CV file not found at: {cv_path}")

with open(cv_path, "r") as f:
    content = f.read()

# Split into chunks (simple split for now)
chunks = content.split("\n\n")

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index + doc map
faiss.write_index(index, index_path)

with open(doc_map_path, "wb") as f:
    pickle.dump({i: chunk for i, chunk in enumerate(chunks)}, f)

print(f"✅ FAISS index saved to {index_path}")
print(f"✅ Document map saved to {doc_map_path}")

