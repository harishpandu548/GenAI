from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

# 1. prompt template
prompt=ChatPromptTemplate.from_template(
    "explain {topic} in simple words"
)

# 2. model
model = ChatMistralAI(model="mistral-small-2506")

# 3. output parser
parser=StrOutputParser()

chain=prompt | model | parser

response=chain.invoke("Movies")

print(response)