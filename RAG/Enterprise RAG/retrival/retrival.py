# # retrieval/retriever.py
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain_community.vectorstores.pgvector import PGVector

# def get_qa_chain(llm, vectorstore: PGVector) -> RetrievalQA:
#     """
#     Create a RetrievalQA chain using given LLM and vectorstore.
#     """
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         chain_type="stuff"
#     )
# retrieval/retriever.py
def get_qa_chain(llm, vectorstore):
    """
    Returns a RetrievalQA chain
    """
    from llm.qa_chain import get_qa_chain as build_qa_chain
    return build_qa_chain(llm, vectorstore)
