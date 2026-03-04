from langchain_community.document_loaders import TextLoader 

loader =TextLoader(r"D:\GENAI\LangChain\RAG\Documents\agentic_ai_sample.txt",encoding="utf-8")
documents=loader.load()
print(f"Numbers of documents loaded :{len(documents)}")
print(documents)