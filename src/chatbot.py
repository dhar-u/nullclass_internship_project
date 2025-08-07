import os
from analytics import log_interaction
import time
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()  # Load HuggingFace API key from .env if present

# Set model name for embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Function to load vector index for a given language
def load_vector_index(language_code):
    index_path = f"./data/{language_code}"
    if not os.path.exists(index_path):
        raise ValueError(f"Vector index not found for language: {language_code}")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Function to ask a question in a specific language
def ask_question(question, language_code):
    vectorstore = load_vector_index(language_code)

    # Use HuggingFace LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # You can change to another model if needed
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    result = qa_chain.run(question)
    return result

# Main script
if __name__ == "__main__":
    print("Welcome to the multilingual chatbot!")
    question = input("Enter your question: ")
    lang = input("Enter language code (en, hi, ta, fr): ")
    
    try:
        answer = ask_question(question, lang)
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
