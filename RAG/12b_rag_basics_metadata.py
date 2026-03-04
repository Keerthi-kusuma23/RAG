# Load API keys safely

# Open an existing vector database

# Convert your question into embeddings

# Find the most similar stored documents

# Give those documents to GPT

# Force GPT to answer only from those documents

# Print the final answer


import os
from dotenv import load_dotenv

# LangChain / Chroma imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables (OPENAI_API_KEY etc.)
load_dotenv()

# --- safe path handling (works in script or notebook) ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ doesn't exist (e.g., notebook), fall back to cwd
    current_dir = os.getcwd()

persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
os.makedirs(persistent_directory, exist_ok=True)

# --- embeddings & Chroma vectorstore ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# --- build retriever ---
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)

query = "Where is Dracula's castle located?"

# Use standard retriever API (get_relevant_documents)
try:
    relevant_docs = retriever.get_relevant_documents(query)
except AttributeError:
    # Some LangChain versions call it 'retrieve'
    relevant_docs = retriever.retrieve(query)

# Print retrieved docs (safely)
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, start=1):
    content = getattr(doc, "page_content", str(doc))
    print(f"Document {i}:\n{content}\n")
    src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
    print(f"Source: {src}\n")

# Combine documents for LLM input (only use contents we actually retrieved)
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([getattr(d, "page_content", "") for d in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. "
    + "If the answer is not found in the documents, respond with 'I'm not sure'."
)

# --- call ChatOpenAI (use predict_messages for a chat-style return) ---
model = ChatOpenAI(model="gpt-4o")  # or model_name="gpt-4o" depending on version

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Prefer predict_messages (returns an AIMessage with .content).
# If predict_messages isn't available in your version, try: model(messages) or model.generate(...)
try:
    response = model.predict_messages(messages)
    assistant_text = response.content
except AttributeError:
    # fallback: call the model directly
    try:
        # model(...) may return an AIMessage or LLMResult depending on version
        resp = model(messages)
        assistant_text = getattr(resp, "content", str(resp))
    except Exception as e:
        # last resort: use generate()
        gen = model.generate([messages])
        # model.generate returns LLMResult; extract text
        assistant_text = gen.generations[0][0].text

print("\n--- Generated Response ---")
print(assistant_text)
