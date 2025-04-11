from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os, pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_dir = "data/docs"
doc_texts = []
for filename in os.listdir(doc_dir):
    with open(os.path.join(doc_dir, filename), 'r') as f:
        doc_texts.append(f.read())

embeddings = model.encode(doc_texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "data/index.faiss")

with open("data/doc_map.pkl", "wb") as f:
    pickle.dump({i: doc for i, doc in enumerate(doc_texts)}, f)