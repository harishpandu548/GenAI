# load pdf
# split into chunks
# create the embeddings
# store into chroma

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

data = PyPDFLoader("RAG_project/document_loader/Harish_CV-3.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30,
)

chunks=splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorStore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)