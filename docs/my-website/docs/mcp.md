import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# /mcp [BETA] - Model Context Protocol

## Expose MCP tools on LiteLLM Proxy Server

This allows you to define tools that can be called by any MCP compatible client. Define your `mcp_servers` with LiteLLM and all your clients can list and call available tools.

<Image 
  img={require('../img/mcp_2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  LiteLLM MCP Architecture: Use MCP tools with all LiteLLM supported models
</p>

#### How it works

1. Allow proxy admin users to perform create, update, and delete operations on MCP servers stored in the db.
2. Allows users to view and call tools to the MCP servers they have access to.

LiteLLM exposes the following MCP endpoints:

- GET `/mcp/enabled` - Returns if MCP is enabled (python>=3.10 requirements are met)
- GET `/mcp/tools/list` - List all available tools
- POST `/mcp/tools/call` - Call a specific tool with the provided arguments
- GET `/v1/mcp/server` - Returns all of the configured mcp servers in the db filtered by requestor's access
- GET `/v1/mcp/server/{server_id}` - Returns the the specific mcp server in the db given `server_id` filtered by requestor's access
- PUT  `/v1/mcp/server` - Updates an existing external mcp server.
- POST `/v1/mcp/server` - Add a new external mcp server.
- DELETE `/v1/mcp/server/{server_id}` - Deletes the mcp server given `server_id`.

When MCP clients connect to LiteLLM they can follow this workflow:

1. Connect to the LiteLLM MCP server
2. List all available tools on LiteLLM
3. Client makes LLM API request with tool call(s)
4. LLM API returns which tools to call and with what arguments
5. MCP client makes MCP tool calls to LiteLLM
6. LiteLLM makes the tool calls to the appropriate MCP server
7. LiteLLM returns the tool call results to the MCP client

#### Usage

#### 1. Define your tools on under `mcp_servers` in your config.yaml file.

LiteLLM allows you to define your tools on the `mcp_servers` section in your config.yaml file. All tools listed here will be available to MCP clients (when they connect to LiteLLM and call `list_tools`).

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

mcp_servers:
  zapier_mcp:
    url: "https://actions.zapier.com/mcp/sk-akxxxxx/sse"
  fetch:
    url: "http://localhost:8000/sse"
```


#### 2. Start LiteLLM Gateway

<Tabs>
<TabItem value="docker" label="Docker Run">

```shell title="Docker Run" showLineNumbers
docker run -d \
  -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --name my-app \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  my-app:latest \
  --config /app/config.yaml \
  --port 4000 \
  --detailed_debug \
```

</TabItem>

<TabItem value="py" label="litellm pip">

```shell title="litellm pip" showLineNumbers
litellm --config config.yaml --detailed_debug
```

</TabItem>
</Tabs>


#### 3. Make an LLM API request 

In this example we will do the following:

1. Use MCP client to list MCP tools on LiteLLM Proxy
2. Use `transform_mcp_tool_to_openai_tool` to convert MCP tools to OpenAI tools
3. Provide the MCP tools to `gpt-4o`
4. Handle tool call from `gpt-4o`
5. Convert OpenAI tool call to MCP tool call
6. Execute tool call on MCP server

```python title="MCP Client List Tools" showLineNumbers
import asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from mcp import ClientSession
from mcp.client.sse import sse_client
from litellm.experimental_mcp_client.tools import (
    transform_mcp_tool_to_openai_tool,
    transform_openai_tool_call_request_to_mcp_tool_call_request,
)


async def main():
    # Initialize clients
    
    # point OpenAI client to LiteLLM Proxy
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://localhost:4000")

    # Point MCP client to LiteLLM Proxy
    async with sse_client("http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List MCP tools on LiteLLM Proxy
            mcp_tools = await session.list_tools()
            print("List of MCP tools for MCP server:", mcp_tools.tools)

            # Create message
            messages = [
                ChatCompletionUserMessageParam(
                    content="Send an email about LiteLLM supporting MCP", role="user"
                )
            ]

            # 2. Use `transform_mcp_tool_to_openai_tool` to convert MCP tools to OpenAI tools
            # Since OpenAI only supports tools in the OpenAI format, we need to convert the MCP tools to the OpenAI format.
            openai_tools = [
                transform_mcp_tool_to_openai_tool(tool) for tool in mcp_tools.tools
            ]

            # 3. Provide the MCP tools to `gpt-4o`
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            # 4. Handle tool call from `gpt-4o`
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call:

                    # 5. Convert OpenAI tool call to MCP tool call
                    # Since MCP servers expect tools in the MCP format, we need to convert the OpenAI tool call to the MCP format.
                    # This is done using litellm.experimental_mcp_client.tools.transform_openai_tool_call_request_to_mcp_tool_call_request
                    mcp_call = (
                        transform_openai_tool_call_request_to_mcp_tool_call_request(
                            openai_tool=tool_call.model_dump()
                        )
                    )

                    # 6. Execute tool call on MCP server
                    result = await session.call_tool(
                        name=mcp_call.name, arguments=mcp_call.arguments
                    )

                    print("Result:", result)


# Run it
asyncio.run(main())
```

## LiteLLM Python SDK MCP Bridge

LiteLLM Python SDK acts as a MCP bridge to utilize MCP tools with all LiteLLM supported models. LiteLLM offers the following features for using MCP

- **List** Available MCP Tools: OpenAI clients can view all available MCP tools
  - `litellm.experimental_mcp_client.load_mcp_tools` to list all available MCP tools
- **Call** MCP Tools: OpenAI clients can call MCP tools
  - `litellm.experimental_mcp_client.call_openai_tool` to call an OpenAI tool on an MCP server


### 1. List Available MCP Tools

In this example we'll use `litellm.experimental_mcp_client.load_mcp_tools` to list all available MCP tools on any MCP server. This method can be used in two ways:

- `format="mcp"` - (default) Return MCP tools 
  - Returns: `mcp.types.Tool`
- `format="openai"` - Return MCP tools converted to OpenAI API compatible tools. Allows using with OpenAI endpoints.
  - Returns: `openai.types.chat.ChatCompletionToolParam`

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python title="MCP Client List Tools" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import litellm
from litellm import experimental_mcp_client


server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=tools,
        )
        print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))
```

</TabItem>

<TabItem value="openai" label="OpenAI SDK + LiteLLM Proxy">

In this example we'll walk through how you can use the OpenAI SDK pointed to the LiteLLM proxy to call MCP tools. The key difference here is we use the OpenAI SDK to make the LLM API request

```python title="MCP Client List Tools" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from openai import OpenAI
from litellm import experimental_mcp_client

server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools using litellm mcp client
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        # Use OpenAI SDK pointed to LiteLLM proxy
        client = OpenAI(
            api_key="your-api-key",  # Your LiteLLM proxy API key
            base_url="http://localhost:4000"  # Your LiteLLM proxy URL
        )

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        print("LLM RESPONSE: ", llm_response)
```
</TabItem>
</Tabs>


### 2. List and Call MCP Tools

In this example we'll use 
- `litellm.experimental_mcp_client.load_mcp_tools` to list all available MCP tools on any MCP server
- `litellm.experimental_mcp_client.call_openai_tool` to call an OpenAI tool on an MCP server

The first llm response returns a list of OpenAI tools. We take the first tool call from the LLM response and pass it to `litellm.experimental_mcp_client.call_openai_tool` to call the tool on the MCP server.

#### How `litellm.experimental_mcp_client.call_openai_tool` works

- Accepts an OpenAI Tool Call from the LLM response
- Converts the OpenAI Tool Call to an MCP Tool
- Calls the MCP Tool on the MCP server
- Returns the result of the MCP Tool call

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python title="MCP Client List and Call Tools" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import litellm
from litellm import experimental_mcp_client


server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=tools,
        )
        print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))

        openai_tool = llm_response["choices"][0]["message"]["tool_calls"][0]
        # Call the tool using MCP client
        call_result = await experimental_mcp_client.call_openai_tool(
            session=session,
            openai_tool=openai_tool,
        )
        print("MCP TOOL CALL RESULT: ", call_result)

        # send the tool result to the LLM
        messages.append(llm_response["choices"][0]["message"])
        messages.append(
            {
                "role": "tool",
                "content": str(call_result.content[0].text),
                "tool_call_id": openai_tool["id"],
            }
        )
        print("final messages with tool result: ", messages)
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=tools,
        )
        print(
            "FINAL LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str)
        )
```

</TabItem>
<TabItem value="proxy" label="OpenAI SDK + LiteLLM Proxy">

In this example we'll walk through how you can use the OpenAI SDK pointed to the LiteLLM proxy to call MCP tools. The key difference here is we use the OpenAI SDK to make the LLM API request

```python title="MCP Client with OpenAI SDK" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from openai import OpenAI
from litellm import experimental_mcp_client

server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools using litellm mcp client
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        # Use OpenAI SDK pointed to LiteLLM proxy
        client = OpenAI(
            api_key="your-api-key",  # Your LiteLLM proxy API key
            base_url="http://localhost:8000"  # Your LiteLLM proxy URL
        )

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        print("LLM RESPONSE: ", llm_response)

        # Get the first tool call
        tool_call = llm_response.choices[0].message.tool_calls[0]
        
        # Call the tool using MCP client
        call_result = await experimental_mcp_client.call_openai_tool(
            session=session,
            openai_tool=tool_call.model_dump(),
        )
        print("MCP TOOL CALL RESULT: ", call_result)

        # Send the tool result back to the LLM
        messages.append(llm_response.choices[0].message.model_dump())
        messages.append({
            "role": "tool",
            "content": str(call_result.content[0].text),
            "tool_call_id": tool_call.id,
        })

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        print("FINAL RESPONSE: ", final_response)
```

</TabItem>
</Tabs>

### Permission Management

Currently, all Virtual Keys are able to access the MCP endpoints. We are working on a feature to allow restricting MCP access by keys/teams/users/orgs.

Join the discussion [here](https://github.com/BerriAI/litellm/discussions/9891)