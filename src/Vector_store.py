from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load the embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths to save files
index_path = "data/index.faiss"
store_path = "data/doc_store.pkl"

def create_index(documents):
    vectors = model.encode(documents)
    dim = vectors.shape[1]

    # Create a FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    # Save the index
    faiss.write_index(index, index_path)

    # Save documents
    doc_store = {i: doc for i, doc in enumerate(documents)}
    with open(store_path, "wb") as f:
        pickle.dump(doc_store, f)

    print("✅ Initial index and doc store created.")


def load_index():
    if not os.path.exists(index_path) or not os.path.exists(store_path):
        raise Exception("Index or document store not found. Run init_index.py first.")

    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        doc_store = pickle.load(f)
    return index, doc_store


def embed_query(query):
    return model.encode([query])
