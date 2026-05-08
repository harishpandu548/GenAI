import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(model="mistral-small-2506")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in a clean readable format.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep output short (2–3 lines max)
"""),
    ("human", "Extract info from this para: {paragraph}")
])

# UI
st.set_page_config(page_title="🎬 Movie Info Extractor", page_icon="🎬")
st.title("🎬 Movie Information Extractor")

st.write("Paste a movie paragraph and get structured info instantly.")

# Input box
paragraph = st.text_area("Enter Movie Paragraph", height=200)

# Button
if st.button("Extract Info"):

    if paragraph.strip() == "":
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting..."):
            final_prompt = prompt.invoke({
                "paragraph": paragraph
            })

            response = model.invoke(final_prompt)

        st.success("Extraction Complete ✅")

        # Output
        st.subheader("📄 Extracted Information")
        st.code(response.content)