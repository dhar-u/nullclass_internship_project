from src.Vector_store import create_index

with open("data/init_docs.txt", "r") as f:
    docs = [line.strip() for line in f.readlines() if line.strip()]

create_index(docs)
