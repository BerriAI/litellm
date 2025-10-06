import React from "react";

const codeString = `import asyncio
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
    client = AsyncOpenAI(
        api_key="sk-1234", 
        base_url="http://localhost:4000"
    )
    
    # Connect to MCP
    async with sse_client("http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            print("List of MCP tools for MCP server:", mcp_tools.tools)
            
            # Create message
            messages = [
                ChatCompletionUserMessageParam(
                    content="Send an email about LiteLLM supporting MCP",
                    role="user"
                )
            ]
            
            # Request with tools
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[transform_mcp_tool_to_openai_tool(tool) for tool in mcp_tools.tools],
                tool_choice="auto"
            )
            
            # Handle tool call
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call:
                    # Convert format
                    mcp_call = transform_openai_tool_call_request_to_mcp_tool_call_request(
                        openai_tool=tool_call.model_dump()
                    )
                    
                    # Execute tool
                    result = await session.call_tool(
                        name=mcp_call.name,
                        arguments=mcp_call.arguments
                    )
                    
                    print("Result:", result)

# Run it
asyncio.run(main())`;

export const CodeExample: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow h-full">
      <div className="border-b px-4 py-3">
        <h3 className="text-base font-medium text-gray-900">Using MCP Tools</h3>
      </div>
      <div className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <div className="flex-1">
            <div className="text-sm font-medium text-gray-700">Python integration</div>
          </div>
          <button
            className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-600 px-2 py-1 rounded-md transition-colors"
            onClick={() => {
              navigator.clipboard.writeText(codeString);
            }}
          >
            Copy
          </button>
        </div>

        <div className="overflow-auto rounded-md bg-gray-50 border" style={{ maxHeight: "calc(100vh - 280px)" }}>
          <pre className="p-3 text-xs font-mono text-gray-800 whitespace-pre overflow-x-auto">{codeString}</pre>
        </div>
      </div>
    </div>
  );
};
