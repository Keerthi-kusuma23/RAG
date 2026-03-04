# from langchain_community.document_loaders.parsers.pdf import PyPDFParser
# from langchain_core.documents.base import Blob
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters.character import RecursiveCharacterTextSplitter
# from langchain_postgres import PGVector
# from langchain_openai.chat_models import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv() # load environment variables from .env file
# connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# model = OpenAIEmbeddings(model="text-embedding-3-small")

# blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

# parser = PyPDFParser()

# documents = parser.lazy_parse(blob)
# docs = []
# for doc in documents:
#     docs.append(doc)
# # print(docs[0].page_content)
# # print(docs[0].metadata)
# # print(docs)

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,  # Adjust chunk size as needed
#     chunk_overlap=50  # Adjust overlap as needed
# )

# chunks = splitter.split_documents(docs)
# # print(f"Number of chunks created: {len(chunks)}")
# # for i, chunk in enumerate(chunks):
# #     print(f"Chunk {i+1}:")
# #     print(chunk.page_content)  # Print the content of each chunk
# #     print("--------------------------------")

# # create vector store
# vector_store = PGVector.from_documents(
#     documents=chunks,
#     embedding=model,
#     connection=connection,
#     use_jsonb=True,  # Use JSONB for metadata storage
# )

# # # Print the vector store
# # print("Vector Store:")
# # print(vector_store)

# retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# query = "What is the experience of Arjun Varma in AI?"

# docs = retriever.invoke(query) # retrieve the context based on the query

# print("Retrieved documents: ")
# for doc in docs:
#     print(doc.page_content)
#     print("--------------------------------")

# prompt = ChatPromptTemplate.from_template(
#     """
#     You are a helpful assistant that answers the details based on the context provided.
#     Context: {context}
#     Question: {question}
#     """
# )
 
# llm = ChatOpenAI(model="gpt-5-nano")

# chain = prompt | llm

# user_input = {"context": docs, "question": query} # pass the context and question to the chain

# response = chain.invoke(user_input)

# print("Response:")
# print(response.content)







from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1️⃣ Load environment variables (for your OpenAI API key)
load_dotenv()

# 2️⃣ Create Embedding model
model = OpenAIEmbeddings(model="text-embedding-3-small")

# 3️⃣ Load and parse PDF file
blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")
parser = PyPDFParser()
documents = parser.lazy_parse(blob)
docs = [doc for doc in documents]

# 4️⃣ Split into smaller chunks (important for vector search)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 5️⃣ Create FAISS vector store (local memory, no DB)
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=model
)

# 6️⃣ Save FAISS index locally (optional)
vector_store.save_local("faiss_index")

# 7️⃣ Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 8️⃣ Ask your query
query = "What is the experience of Arjun Varma in AI?"

docs = retriever.invoke(query)  # Retrieve the most relevant chunks

print("Retrieved documents:")
for doc in docs:
    print(doc.page_content)
    print("--------------------------------")

# 9️⃣ Create prompt template for LLM
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers the details based on the context provided.
    Context: {context}
    Question: {question}
    """
)

# 🔟 Initialize the Chat Model (LLM)
llm = ChatOpenAI(model="gpt-5-nano")

# 11️⃣ Combine prompt + model into a chain
chain = prompt | llm

# 12️⃣ Run the chain with context and question
user_input = {"context": docs, "question": query}
response = chain.invoke(user_input)

print("Response:")
print(response.content)











