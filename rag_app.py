import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF RAG", page_icon="📄")
st.title("📄 Chat with your PDF")

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---------------- Process PDF (YOUR LOGIC) ----------------
if uploaded_file:
    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            # save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            # ---- YOUR created_db_for_main.py LOGIC ----
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=30,
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # IMPORTANT → Persist to disk (same as your code)
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            st.success("PDF processed and stored in Chroma DB ✅")

# ---------------- Chat (YOUR main.py LOGIC) ----------------
query = st.text_input("Ask your question")

if query:

    # load DB from disk (same as your main.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an helpful AI assistant. Use only the provided context to answer the question.
If the answer is not provided in the context just say: I could not find the solution in the document
"""
        ),
        (
            "human",
            "context:{context}\nquestion:{question}"
        )
    ])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    llm = ChatMistralAI(model="mistral-small-2506")
    response = llm.invoke(final_prompt)

    st.write("### 🤖 Answer")
    st.write(response.content) 