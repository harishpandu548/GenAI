from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]


vectorStore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma-db"
)

# print(vectorStore)

result = vectorStore.similarity_search("what is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

print(".......................")

retriever=vectorStore.as_retriever()

docs=retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
    print(r.metadata)