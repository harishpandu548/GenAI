# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# from langchain_mistralai import ChatMistralAI

# model = ChatMistralAI(model='mistral-small-2506')

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
# You are a professional Movie Information Extraction Assistant.

# Your task:
# Extract useful structured information from a movie paragraph and present it in a clean readable format.

# Rules:
# - Do NOT add explanations
# - Do NOT add extra commentary
# - Follow the exact format
# - If information is missing → write NULL
# - Keep output short (2–3 lines max)
# """),
# ("human","""
# Extract info from this para: {paragraph}""")
# ])

# para = input("Give your paragraph : ")

# final_prompt = prompt.invoke(
#     {"paragraph" : para
#      }
# )

# response = model.invoke(final_prompt)

# print(response.content)

# ---------------------------------------------------------------------------------------------------
# Example of using pydantic 

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = 'mistral-small-2506')

class Movie(BaseModel):
    title: str 
    release_year : Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ('system',"""
Extract movie information from the paragraph
     {format_instructions}
"""),
("human","{paragraph}")
]
)

para = input("Give your paragraph : ")

final_prompt = prompt.invoke(
    {"paragraph" : para,
     'format_instructions': parser.get_format_instructions()
     }
)

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)

print(movie_data)
