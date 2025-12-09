from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
from openai import OpenAI

# --- CONFIG ---
# Point LangChain to your LiteLLM proxy server
os.environ["OPENAI_API_BASE"] = "http://localhost:4000"  # update if needed

# --- LLM ---
# Use gpt-4o-mini model through LiteLLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# --- Example Tool ---
@tool
def say_hello(name: str) -> str:
    """Tool to say hello or greet someone"""
    return f"Hello, {name}!"


# --- Agent ---
agent = create_agent(
    model=llm,
    tools=[say_hello],
)

# --- Run ---
response = agent.invoke({"messages": [{"role": "user", "content": "Use the say_hello tool to greet ashton"}]})
print(response)
