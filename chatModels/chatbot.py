from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

model = ChatMistralAI(model="mistral-small-2506") 

print("Welcome Back, Press 0 to exit the chat")

# temporary chat history so model remembers your context. As we know this models api req are stateless
messages=[
    SystemMessage(content="You are a very funny AI agent")
]

while True:
    
    prompt=input("You : ")
    messages.append(HumanMessage(content=prompt))

    if(prompt=="0"):
        break

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot : ",response.content)

print(messages)