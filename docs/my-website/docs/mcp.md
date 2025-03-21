import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /mcp Model Context Protocol [Beta]

Use Model Context Protocol with LiteLLM. 

## Overview

LiteLLM supports Model Context Protocol (MCP) tools by offering a client that exposes a tools method for retrieving tools from a MCP server

## Usage

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import asyncio
from litellm import experimental_create_mcp_client, completion
from litellm.mcp_stdio import experimental_stdio_mcp_transport

async def main():
    client_one = None

    try:
        # Initialize an MCP client to connect to a `stdio` MCP server:
        transport = experimental_stdio_mcp_transport(
            command='node',
            args=['src/stdio/dist/server.js']
        )
        client_one = await experimental_create_mcp_client(
            transport=transport
        )

        tools = await client_one.list_tools(format="openai")
        response = await litellm.completion(
            model="gpt-4o",
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": "Find products under $100"
                }
            ]
        )

        print(response.text)
    except Exception as error:
        print(error)
    finally:
        await asyncio.gather(
            client_one.close() if client_one else asyncio.sleep(0),
        )

if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy Server">

```python
import asyncio
from openai import OpenAI
from litellm import experimental_create_mcp_client
from litellm.mcp_stdio import experimental_stdio_mcp_transport

async def main():
    client_one = None

    try:
        # Initialize an MCP client to connect to a `stdio` MCP server:
        transport = experimental_stdio_mcp_transport(
            command='node',
            args=['src/stdio/dist/server.js']
        )
        client_one = await experimental_create_mcp_client(
            transport=transport
        )

        # Get tools from MCP client
        tools = await client_one.list_tools(format="openai")
        
        # Use OpenAI client connected to LiteLLM Proxy Server
        client = openai.OpenAI(
            api_key="sk-1234",
            base_url="http://0.0.0.0:4000"
        )
        response = client.chat.completions.create(
            model="gpt-4",
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": "Find products under $100"
                }
            ]
        )

        print(response.choices[0].message.content)
    except Exception as error:
        print(error)
    finally:
        await asyncio.gather(
            client_one.close() if client_one else asyncio.sleep(0),
        )

if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>
</Tabs>

