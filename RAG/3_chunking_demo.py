from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
#from langchain_text_splitters.character import ReccursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter



blob=Blob.from_path(r"C:\Users\keert\Downloads\hritik_shrivastava_resume.pdf")
parser=PyPDFParser()
documents=parser.lazy_parse(blob)
docs=[]

for doc in documents:
    docs.append(doc)
print(docs)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200

)

chunks=splitter.split_documents(docs)
print(f"numbers of chunks created:{len(chunks)}")
for i, chunks in enumerate(chunks):
    print(f"Chunk{i+i}:")
    print(chunks.page_content)
    print("---------------------------------------------------------------------------------------------------------------------")
