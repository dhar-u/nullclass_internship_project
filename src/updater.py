from src.vector_store import model
import faiss
import numpy as np
import pickle
import os
import schedule
import time

index_path = "data/index.faiss"
store_path = "data/doc_store.pkl"
new_file_path = "data/new_docs.txt"

def update_knowledge():
    # Check if new content file exists
    if not os.path.exists(new_file_path):
        print("🟡 No new_docs.txt found.")
        return

    with open(new_file_path, "r") as f:
        new_docs = [line.strip() for line in f if line.strip()]

    if not new_docs:
        print("🟡 No new documents to add.")
        return

    # Load existing index and doc store
    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        doc_store = pickle.load(f)

    # Embed new documents
    vectors = model.encode(new_docs)
    index.add(np.array(vectors))

    # Update document store with new IDs
    start_id = len(doc_store)
    for i, doc in enumerate(new_docs):
        doc_store[start_id + i] = doc

    # Save updated index and doc store
    faiss.write_index(index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump(doc_store, f)

    print(f"✅ Added {len(new_docs)} new documents to knowledge base.")

def run_scheduler():
    schedule.every(1).minutes.do(update_knowledge)  # You can change to hours

    print("⏳ Auto-updater is running... (checks every 1 minute)")
    while True:
        schedule.run_pending()
        time.sleep(5)

if __name__ == "__main__":
    run_scheduler()
