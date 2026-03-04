from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob

blob=Blob.from_path(r"C:\Users\keert\Downloads\hritik_shrivastava_resume.pdf")
parser=PyPDFParser()
documents=parser.lazy_parse(blob)
docs=[]

for doc in documents:
    docs.append(doc)
print(docs)