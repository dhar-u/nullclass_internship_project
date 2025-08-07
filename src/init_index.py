from vector_store import create_index_for_language

# List of language codes you want to index
languages = ["en", "hi", "ta", "fr"]

for lang in languages:
    create_index_for_language(lang)
