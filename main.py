from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# loading chroma db which was created using created_db_for_main.py here
vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistral-small-2506")

# prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system","""
You are an helpful AI assistant. Use only the provided context to answer the question.
If the answer is not provided in the context just say: I could not find the solution in the document
"""
        ),(
            "human","""
context:{context}
question:{question}
"""
)
    ]
)

print("Rag system created")

print("0 to exit")

while True:
    query=input("you: ")

    if(query=="0"):
        break

    docs=retriever.invoke(query)

    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    response=llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")