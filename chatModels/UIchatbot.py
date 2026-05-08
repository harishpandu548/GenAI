import streamlit as st
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Initialize model
model = ChatMistralAI(model="mistral-small-2506")

# Streamlit UI
st.set_page_config(page_title="Funny AI Chatbot", page_icon="😂")
st.title("😂 Funny AI Chatbot")

# Initialize session state (chat memory)
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a very funny AI agent")
    ]

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    # Get response
    response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.write(response.content)