from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

texts=[
    "Hi",
    "Hello",
    "Hola"
]

vector = embeddings.embed_documents(texts)
# or 
# vector = embeddings.embed_query("hello, world!") if the sentence is of query on simple single sentence use this or else if it is consisting of multiple lines etc use upper one

print(vector)