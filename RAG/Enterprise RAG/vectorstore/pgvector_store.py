# # vectorstore/pgvector_store.py
# import os
# from langchain_community.vectorstores import PGVector
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

# def create_vectorstore(documents):
#     """
#     Create a PGVector vectorstore from documents using either OpenAI or Ollama embeddings.
#     """
#     provider = os.getenv("PROVIDER", "openai").lower()

#     if provider == "ollama":
#         model_name = os.getenv("OLLAMA_MODEL", "nomic-embed-text")  # Ollama model
#         try:
#             embeddings = OllamaEmbeddings(model=model_name)
#         except Exception as e:
#             print(f"Ollama embedding error: {e}. Falling back to OpenAI embeddings.")
#             embeddings = OpenAIEmbeddings()
#     else:
#         embeddings = OpenAIEmbeddings()

#     vectorstore = PGVector.from_documents(
#         documents,
#         embedding=embeddings,
#         collection_name=os.getenv("PGVECTOR_COLLECTION", "enterprise_rag"),
#         connection_string=os.getenv(
#             "PGVECTOR_CONN",
#             "postgresql://langchain:langchain@localhost:5432/langchain"
#         ),
#     )
#     return vectorstore
# vectorstore/pgvector_store.py
# vectorstore/pgvector_store.py
# vectorstore/pgvector_store.py
# vectorstore/pgvector_store.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()   # 👈 ADD THIS LINE ONLY


def create_vectorstore(docs):
    """
    Creates a PGVector vectorstore using Ollama embeddings.
    """

    if not docs:
        raise ValueError("No documents provided for vectorstore")

    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_MODEL", "nomic-embed-text")
    )

    vectorstore = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION", "enterprise_rag"),
        connection_string=os.getenv("PGVECTOR_CONN")  # 👈 NOW THIS WORKS
    )

    return vectorstore


 

