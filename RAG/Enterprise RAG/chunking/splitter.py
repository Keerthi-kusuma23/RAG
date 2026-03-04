# chunking/splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

def split_docs(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into chunks for better embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    for doc in docs:
        texts = text_splitter.split_text(doc.page_content)
        for i, t in enumerate(texts):
            chunks.append(Document(page_content=t, metadata={"source": doc.metadata.get("page", None), "chunk": i + 1}))
    return chunks
