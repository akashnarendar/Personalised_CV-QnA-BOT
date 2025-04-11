from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os, pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

cv_path = "data/docs/Narendar_Punithan_CV_final_word.txt"
with open(cv_path, "r") as f:
    raw_text = f.read()

# Split by double newlines and clean each chunk
raw_chunks = raw_text.split("\n\n")
doc_chunks = [chunk.strip().replace("\n", " ") for chunk in raw_chunks if len(chunk.strip()) > 50]

print(f"[INFO] Loaded {len(doc_chunks)} meaningful CV chunks.")

# Create vector embeddings
embeddings = model.encode(doc_chunks)

# Initialize FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index + mapping
faiss.write_index(index, "data/index.faiss")
with open("data/doc_map.pkl", "wb") as f:
    pickle.dump({i: chunk for i, chunk in enumerate(doc_chunks)}, f)

print("[INFO] FAISS index and doc map saved successfully.")
