from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv() # load environment variables from .env file

model = OpenAIEmbeddings(model="text-embedding-3-small")

blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

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

# Generate embeddings for the chunks
embeddings = model.embed_documents([c.page_content for c in chunks])

# Print the embeddings
print("Embeddings:")
for (i, emb), c in zip (enumerate(embeddings),chunks):
    print(f"For the text: {c.page_content}")
    print(f"Total Dimensions: {len(emb)} Chunk {i+1} Embedding: {emb[:10]}...")  # Print first 10 dimensions of the embedding for brevity
    print("--------------------------------")