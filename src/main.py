from src.Vector_store import load_index, embed_query
import numpy as np

# Load vector DB and document store
index, doc_store = load_index()

def chat():
    print("🤖 Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        query_vec = embed_query(query)
        _, I = index.search(np.array(query_vec), k=1)
        answer = doc_store.get(I[0][0], "Sorry, I don't know the answer.")
        print("🤖 Bot:", answer)

if __name__ == "__main__":
    chat()
