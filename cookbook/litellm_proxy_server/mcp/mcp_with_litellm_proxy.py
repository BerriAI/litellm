"""
Use LiteLLM Proxy MCP Gateway to call MCP tools.

When using LiteLLM Proxy, you can use the same MCP tools across all your LLM providers.
"""
import openai

client = openai.OpenAI(
    api_key="sk-1234", # paste your litellm proxy api key here
    base_url="http://localhost:4000" # paste your litellm proxy base url here
)
print("Making API request to Responses API with MCP tools")

response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": "give me TLDR of what BerriAI/litellm repo is about",
            "type": "message"
        }
    ],
    tools=[
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        }
    ],
    stream=True,
    tool_choice="required"
)

for chunk in response:
    print("response chunk: ", chunk)
