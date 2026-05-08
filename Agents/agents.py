from dotenv import load_dotenv
load_dotenv()

import os 

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage,ToolCall
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
import requests
from langchain.agents.middleware import wrap_tool_call

# weather tool(open weather)
@tool
def get_weather(city:str)->str:
    """get current weather of the city"""

    api_key=os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response=requests.get(url)
    data=response.json()

    temp=data["main"]["temp"]
    desc=data["weather"][0]["description"]

    return f"weather in {city}: {desc}, {temp}"

# news tool (tavily)
tavily_client= TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
    """get latest news about the city"""

    response= tavily_client.search(
        query=f"Latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    output= response.get("results",[])

    if not output:
        return f"No news found for {city}"
    
    news_list=[]

    for r in output:
        title=r.get("title","No title")
        url=r.get("url","")
        snippet=r.get("content","")

        news_list.append(
            f"--{title}\n -{url}\n -{snippet[:100]}"
        )
    
    return f"Latest news in {city}:\n\n"+"\n\n".join(news_list)

# llm setup
llm=ChatMistralAI(model="mistral-small-2506")

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human approval before every tool call"""
    tool_name=request.tool_call["name"]
    confirm=input(f"Agent wants to call '{tool_name}'. Approve yes/no")

    if confirm.lower()!="yes":
        return ToolMessage(
            content="Tool call denied by user",
        )
    return handler(request)

agent = create_agent(
    llm,
    tools=[get_news,get_weather],
    system_prompt="You are helpful city assistant",
    middleware=[human_approval]
)

print("City Agent | Type quit to exit")

while True:
    user_input=input("You: ")
    if user_input.lower()=="quit":
        break
    result=agent.invoke({
        "messages":[{"role":"user","content":user_input}]
    })

    print("bot: ", result["messages"][-1].content)