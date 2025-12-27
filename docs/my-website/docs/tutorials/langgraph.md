import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using LiteLLM with LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs. When combined with LiteLLM, you can leverage LiteLLM's unified interface to access 100+ LLMs while building sophisticated agentic workflows.

:::tip
Check out this comprehensive tutorial on integrating LiteLLM with LangGraph:

📹 **[LiteLLM + LangGraph Tutorial](https://screen.studio/share/Kise7QGQ)**
:::

## Pre-Requisites

```shell
pip install langchain-litellm langchain litellm langchain_openai
```

## Quick Start

### LangGraph Agent with LiteLLM SDK

<Tabs>
<TabItem value="openai" label="OpenAI (GPT)">

```python
import os
from langgraph.graph import StateGraph, START, END
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
import operator

llm = ChatLiteLLM(
    model="gpt-4o",
)

class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]

def llm_call(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)
workflow.add_node("llm", llm_call)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

agent = workflow.compile()

# Run the agent
result = agent.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})
print(result["messages"][-1].content)
```

</TabItem>

<TabItem value="anthropic" label="Anthropic (Claude)">

```python
import os
from langgraph.graph import StateGraph, START, END
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
import operator

llm = ChatLiteLLM(
    model="claude-3-sonnet-20240229",
)

class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]

def llm_call(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)
workflow.add_node("llm", llm_call)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

agent = workflow.compile()

# Run the agent
result = agent.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})
print(result["messages"][-1].content)
```

</TabItem>

<TabItem value="gemini" label="Google (Gemini)">

```python
import os
from langgraph.graph import StateGraph, START, END
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
import operator

# Using Google's Gemini model with ChatLiteLLM
llm = ChatLiteLLM(
    model="gemini-1.5-pro",
)

class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]

def llm_call(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)
workflow.add_node("llm", llm_call)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

agent = workflow.compile()

# Run the agent
result = agent.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]})
print(result["messages"][-1].content)
```

</TabItem>
</Tabs>

## Using LiteLLM Proxy with LangGraph

The LiteLLM Proxy provides a centralized way to manage models, authentication, and observability. This is especially useful for LangGraph applications.

### Setup LiteLLM Proxy

```yaml title="config.yaml"
model_list:
  - model_name: gpt-5-codex
    litellm_params:
      model: openai/gpt-5-codex
      api_key: sk-xxx
      
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: sk-ant-xxx
      
  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-pro
      api_key: xxx

general_settings:
  master_key: sk-1234  # Your master key for authentication
  database_url: postgres://
```

Start the proxy:
```bash
litellm --config config.yaml --port 4000
```

<Tabs>
<TabItem value="chatlitellm" label="ChatLiteLLM with Tools">

```python
import os
from langgraph.graph import StateGraph, START, END
from langchain_litellm import ChatLiteLLM
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import cast
import operator

llm = ChatLiteLLM(
    model="gpt-5-codex",  # model name from your proxy config
    api_base="http://localhost:4000",  # proxy endpoint
    api_key="sk-1234",  # proxy master key
)

# Define tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

tools = [add, get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int

# Define LLM node
def llm_call(state: MessagesState):
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Define tool node
def tool_node(state: MessagesState):
    result = []
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    if tool_calls:
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            obs = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and getattr(last, "tool_calls", None):
        return "tool_node"
    return END

# Build the graph
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Run the agent
messages = [HumanMessage(content="What's the weather in San Francisco? Also, what is 5 + 3?")]
initial_state = cast(MessagesState, {"messages": messages, "llm_calls": 0})
resp = agent.invoke(initial_state)
for m in resp["messages"]:
    print(m)
```

</TabItem>

<TabItem value="chatopenai" label="ChatOpenAI with Tools">

```python
import os
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import cast
import operator

model = ChatOpenAI(
    base_url="http://localhost:4000",  # litellm proxy base url
    api_key="sk-1234",  # proxy master key
    model="gpt-5-codex",  # model name from proxy config
)

# Define tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

tools = [add, get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int

# Define LLM node
def llm_call(state: MessagesState):
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Define tool node
def tool_node(state: MessagesState):
    result = []
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    if tool_calls:
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            obs = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and getattr(last, "tool_calls", None):
        return "tool_node"
    return END

# Build the graph
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Run the agent
messages = [HumanMessage(content="What's the weather in San Francisco? Also, what is 5 + 3?")]
initial_state = cast(MessagesState, {"messages": messages, "llm_calls": 0})
resp = agent.invoke(initial_state)
for m in resp["messages"]:
    print(m)
```

</TabItem>

</Tabs>
