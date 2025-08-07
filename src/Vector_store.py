# src/vector_store.py

# src/vector_store.py

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
VECTOR_DB_DIR = "vectorstore"

def load_documents(language):
    lang_folder = os.path.join(DATA_FOLDER, language)
    docs = []
    for file_name in os.listdir(lang_folder):
        if file_name.endswith(".txt"):
            loader = TextLoader(os.path.join(lang_folder, file_name), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def create_index_for_language(language):
    print(f"Loading documents for language: {language}")
    documents = load_documents(language)
    print(f"[INFO] Loaded {len(documents)} documents.")


    if not documents:
        raise ValueError("No valid document content to index. Please check your .txt files.")

    print("Splitting documents...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("Embedding and creating vector index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    lang_index_dir = os.path.join(VECTOR_DB_DIR, language)
    os.makedirs(lang_index_dir, exist_ok=True)
    vectorstore.save_local(lang_index_dir)
    print(f"Vector index created for {language} and saved to {lang_index_dir}")
