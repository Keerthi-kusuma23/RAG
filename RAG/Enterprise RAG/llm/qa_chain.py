# # llm/qa_chain.py
# import os
# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA

# # Load .env file so environment variables are available
# load_dotenv()

# def get_llm():
#     """
#     Get LLM based on provider environment variable.
#     Supports OpenAI.
#     """
#     provider = os.getenv("PROVIDER", "openai").lower()

#     if provider == "openai":
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment. Add it to .env or pass it.")
        
#         llm = ChatOpenAI(
#             model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
#             temperature=float(os.getenv("OPENAI_TEMP", 0)),
#             openai_api_key=api_key  # ✅ Explicitly pass the key
#         )
#     else:
#         # Placeholder for future LLM providers
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

#     return llm

# def get_qa_chain(llm, vectorstore):
#     """
#     Build a RetrievalQA chain
#     """
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         chain_type="stuff"
#     )
# llm/qa_chain.py
import os

from langchain_community.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA

def get_llm():
    """
    Get LLM based on provider environment variable.
    Currently only OpenAI for question-answering.
    """
    provider = os.getenv("PROVIDER", "openai").lower()

    if provider == "openai":
        return ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("OPENAI_TEMP", 0)),
            openai_api_key=os.getenv("OPENAI_API_KEY")  # fix missing key
        )
    else:
        # Default fallback
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def get_qa_chain(llm, vectorstore):
    """
    Build a RetrievalQA chain
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )
