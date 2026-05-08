from dotenv import load_dotenv
load_dotenv()

#1. Example for init_chat_model

# from langchain.chat_models import init_chat_model

# model=init_chat_model("google_genai:gemini-2.5-flash-lite")

# response=model.invoke("What is IPL ?")

# print(response.content)

#2. Example for Model class

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") #temperature=0,max_tokens=20 we can add them also after model

response = model.invoke("What is IPL?")

print(response.content)