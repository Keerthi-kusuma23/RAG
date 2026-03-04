from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

##fixed format (a standard pattern) used by SQLAlchemy to describe database connection URLs.
# Formula:dialect+driver://username:password@host:port/database_name
#This is called a SQLAlchemy Connection String.
#psycopg2 is a PostgreSQL database adapter for Python.
connection="postgresql+psycopg2://langchain:langchain@localhost:6024/langchain"


model = OpenAIEmbeddings(model="text-embedding-3-small")

blob = Blob.from_path(r"E:\Gen AI Learning\LangChain\langchain1\hritik_shrivastava_resume.pdf")##name 

parser = PyPDFParser()

documents = parser.lazy_parse(blob)
docs = []
for doc in documents:
    docs.append(doc)
# print(docs[0].page_content)
# print(docs[0].metadata)
print(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Adjust chunk size as needed
    chunk_overlap=50  # Adjust overlap as needed
)

chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)  # Print the content of each chunk
    print("--------------------------------")

# create vector store
vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=model,
    connection=connection,
)

# Print the vector store
print("Vector Store:")
print(vector_store)

# Query the vector store
query = "AI engineer"

results = vector_store.similarity_search(query, k=3)  # Get top 3 results
print(f"Results for query '{query}':")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print("--------------------------------")


results = vector_store.similarity_search_with_score(query, k=3)  # Get top 3 results
print(f"Results for query '{query}':")
for i, (result,score) in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print(f"Score: {score}")
    print("--------------------------------")

##FIRST RUN POSTGRESQL WITH PGVECTOR DOCKER CONTAINER to create database inside a docker
    #docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
