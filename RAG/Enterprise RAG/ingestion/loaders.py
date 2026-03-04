# # ingestion/loaders.py
# from typing import List
# from pypdf import PdfReader
# from langchain.docstore.document import Document

# def load_pdf(file) -> List[Document]:
#     """
#     Load a PDF file and return a list of LangChain Document objects.
#     """
#     reader = PdfReader(file)
#     docs = []
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             docs.append(Document(page_content=text, metadata={"page": i + 1}))
#     return docs

# ingestion/loaders.py
from pypdf import PdfReader
from langchain.docstore.document import Document

def load_pdf(file):
    """
    Load PDF and return LangChain Document list
    """
    pdf = PdfReader(file)
    docs = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs









# import os
# from typing import List

# from langchain.docstore.document import Document
# from langchain_core.documents.base import Blob
# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders.parsers.pdf import PyPDFParser
# from docx import Document as DocxDocument

# from pdf2image import convert_from_path
# import pytesseract


# # ---------- OCR FALLBACK ----------
# def ocr_pdf(file_path: str) -> List[Document]:
#     """
#     OCR fallback for scanned PDFs
#     """
#     images = convert_from_path(file_path)
#     docs = []

#     for page_num, image in enumerate(images, start=1):
#         text = pytesseract.image_to_string(image)
#         if text.strip():
#             docs.append(
#                 Document(
#                     page_content=text.strip(),
#                     metadata={
#                         "source": file_path,
#                         "file_type": "pdf",
#                         "page": page_num,
#                         "ocr": True
#                     }
#                 )
#             )
#     return docs


# # ---------- PDF LOADER ----------
# def load_pdf(file_path: str) -> List[Document]:
#     """
#     Production-grade PDF loader using Blob + Parser
#     with OCR fallback for scanned PDFs
#     """
#     blob = Blob.from_path(file_path)
#     parser = PyPDFParser()

#     docs = []

#     # First attempt: normal text extraction
#     for doc in parser.lazy_parse(blob):
#         if doc.page_content and doc.page_content.strip():
#             doc.metadata["source"] = file_path
#             doc.metadata["file_type"] = "pdf"
#             doc.metadata["ocr"] = False
#             docs.append(doc)

#     # Fallback to OCR if no text found
#     if not docs:
#         docs = ocr_pdf(file_path)

#     return docs


# # ---------- TXT LOADER ----------
# def load_txt(file_path: str) -> List[Document]:
#     """
#     Load TXT file using LangChain TextLoader
#     """
#     loader = TextLoader(file_path, encoding="utf-8")
#     docs = loader.load()

#     for doc in docs:
#         doc.metadata["source"] = file_path
#         doc.metadata["file_type"] = "txt"

#     return docs


# # ---------- DOCX LOADER ----------
# def load_docx(file_path: str) -> List[Document]:
#     """
#     Load DOCX file and convert to LangChain Document
#     """
#     doc = DocxDocument(file_path)
#     text_blocks = []

#     for para in doc.paragraphs:
#         if para.text.strip():
#             text_blocks.append(para.text.strip())

#     text = "\n".join(text_blocks)

#     return [
#         Document(
#             page_content=text,
#             metadata={
#                 "source": file_path,
#                 "file_type": "docx"
#             }
#         )
#     ]


# # ---------- ENTRY POINT ----------
# def load_document(file_path: str) -> List[Document]:
#     """
#     Auto-detect file type and load accordingly
#     """
#     ext = os.path.splitext(file_path)[-1].lower()

#     if ext == ".pdf":
#         return load_pdf(file_path)
#     elif ext == ".txt":
#         return load_txt(file_path)
#     elif ext == ".docx":
#         return load_docx(file_path)
#     else:
#         raise ValueError(f"Unsupported file type: {ext}")
