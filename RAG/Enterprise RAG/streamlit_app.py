# # # streamlit_app.py
# # import os
# # import streamlit as st
# # from ingestion.loaders import load_pdf
# # from chunking.splitter import split_docs
# # from vectorstore.pgvector_store import create_vectorstore
# # from llm.qa_chain import get_llm, get_qa_chain

# # # Page setup
# # st.set_page_config(page_title="Enterprise RAG App", layout="wide")
# # st.title("Enterprise RAG App (OpenAI & Ollama)")

# # # Provider selector
# # provider = st.selectbox("Select Embedding/LLM Provider", ["openai", "ollama"])
# # os.environ["PROVIDER"] = provider
# # st.info(f"Using provider: {provider.upper()}")

# # # Upload PDF
# # uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# # if uploaded_file:
# #     docs = load_pdf(uploaded_file)
# #     st.success(f"Loaded {len(docs)} document(s).")

# #     chunks = split_docs(docs)
# #     st.success(f"Split into {len(chunks)} chunks.")

# #     try:
# #         vectorstore = create_vectorstore(chunks)
# #         st.success("Vectorstore created successfully!")
# #     except Exception as e:
# #         st.error(f"Error creating vectorstore: {e}")

# #     llm = get_llm()
# #     st.info(f"Using LLM model: {os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')}")

# #     query = st.text_input("Ask something about your document:")
# #     if query:
# #         qa_chain = get_qa_chain(llm, vectorstore)
# #         answer = qa_chain.run(query)
# #         st.write("Answer:", answer)
# # streamlit_app.py
# import os
# import streamlit as st
# from ingestion.loaders import load_pdf
# from chunking.splitter import split_docs
# from vectorstore.pgvector_store import create_vectorstore
# from llm.qa_chain import get_llm, get_qa_chain

# st.set_page_config(page_title="Enterprise RAG App", layout="wide")
# st.title("Enterprise RAG App (OpenAI & Ollama)")

# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# query = st.text_input("Ask something about your document:")

# vectorstore = None  # default

# if uploaded_file:
#     docs = load_pdf(uploaded_file)
#     st.success(f"Loaded {len(docs)} page(s) from PDF.")

#     chunks = split_docs(docs)
#     st.success(f"Split into {len(chunks)} chunks.")

#     try:
#         vectorstore = create_vectorstore(chunks)
#         st.success("Vectorstore created!")
#     except Exception as e:
#         st.error(f"Vectorstore creation failed: {e}")

# llm = get_llm()
# st.info(f"Using LLM Provider: {os.getenv('PROVIDER', 'openai')}")

# if vectorstore and query:
#     try:
#         qa_chain = get_qa_chain(llm, vectorstore)
#         answer = qa_chain.run(query)
#         st.write("Answer:", answer)
#     except Exception as e:
#         st.error(f"QA chain failed: {e}")
# streamlit_app.py
import os
from dotenv import load_dotenv
import streamlit as st

# --- LOAD .env ---
load_dotenv()  # this ensures all env variables are available

# --- IMPORT YOUR MODULES ---
from ingestion.loaders import load_pdf
from chunking.splitter import split_docs
from vectorstore.pgvector_store import create_vectorstore
from llm.qa_chain import get_llm, get_qa_chain

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="Enterprise RAG App", layout="wide")
st.title("Enterprise RAG App (OpenAI & Ollama)")

# --- SHOW WHICH PROVIDER IS ACTIVE ---
provider = os.getenv("PROVIDER", "openai").lower()
st.info(f"Active LLM Provider: {provider.upper()}")

# --- UPLOAD PDF ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    docs = load_pdf(uploaded_file)
    st.success(f"Loaded {len(docs)} document(s) from PDF.")

    # --- SPLIT DOCUMENTS INTO CHUNKS ---
    chunks = split_docs(docs)
    st.success(f"Split into {len(chunks)} chunks.")

    # --- CREATE VECTORSTORE ---
    try:
        vectorstore = create_vectorstore(chunks)
        st.success("Vectorstore created!")
    except Exception as e:
        st.error(f"Vectorstore creation failed: {e}")
        st.stop()  # stop execution if vectorstore fails

    # --- INITIALIZE LLM ---
    try:
        llm = get_llm()
    except Exception as e:
        st.error(f"LLM initialization failed: {e}")
        st.stop()

    # --- USER QUERY ---
    query = st.text_input("Ask something about your document:")
    if query:
        try:
            qa_chain = get_qa_chain(llm, vectorstore)
            answer = qa_chain.run(query)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error during Q&A: {e}")
