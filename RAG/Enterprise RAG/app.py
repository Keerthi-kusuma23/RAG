import os
from dotenv import load_dotenv

from ingestion.loaders import load_pdf
from chunking.splitter import split_docs
from vectorstore.pgvector_store import get_vectorstore

from llm.qa_chain import get_qa_chain
from evalution.metrics import is_grounded
from retrival.retrival import get_retriever


load_dotenv()

CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    docs = []

    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            docs.extend(load_pdf(f"docs/{file}"))

    chunks = split_docs(docs)
    vectorstore = create_vectorstore(chunks, CONNECTION)

    retriever = get_retriever(vectorstore)
    qa_chain = get_qa_chain()

    while True:
        query = input("Ask question (type 'exit'): ")
        if query.lower() == "exit":
            break

        retrieved_docs = retriever.invoke(query)
        context = format_docs(retrieved_docs)

        response = qa_chain.invoke({
            "context": context,
            "question": query
        })

        print("\nAnswer:")
        print(response.content)
        print("Grounded:", is_grounded(response.content))
        print("-" * 60)

if __name__ == "__main__":
    main()
