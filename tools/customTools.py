from langchain.tools import tool

@tool
def get_greeting(name:str):
    """Generate a greeting message for a user"""

    return f"Hello {name}, hello world"

result=get_greeting.invoke({"name":"harish"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)