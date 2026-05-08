from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

data = TextLoader("RAG_project/document_loader/Text-to-PDF-RMc.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30,
)

chunks=splitter.split_documents(docs)

template=ChatPromptTemplate.from_messages([
    ("system","You are an AI that summarizes the text"),
    ("human","{data}")
])

model = ChatMistralAI(model="mistral-small-2506") 

prompt=template.format_messages(data=docs[0].page_content)

response = model.invoke(prompt)

print(response.content) 