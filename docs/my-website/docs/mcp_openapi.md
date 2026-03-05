import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MCP from OpenAPI Specs

LiteLLM can automatically convert OpenAPI specifications into MCP servers, exposing any REST API as MCP tools without writing custom MCP server code.

## Adding an OpenAPI MCP Server

Add your OpenAPI-based MCP server in `config.yaml`:

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  # OpenAPI Spec Example - Petstore API
  petstore_mcp:
    url: "https://petstore.swagger.io/v2"
    spec_path: "/path/to/openapi.json"
    auth_type: "none"

  # OpenAPI Spec with API Key Authentication
  my_api_mcp:
    url: "http://0.0.0.0:8090"
    spec_path: "/path/to/openapi.json"
    auth_type: "api_key"
    auth_value: "your-api-key-here"

  # OpenAPI Spec with Bearer Token
  secured_api_mcp:
    url: "https://api.example.com"
    spec_path: "/path/to/openapi.json"
    auth_type: "bearer_token"
    auth_value: "your-bearer-token"
```

**Configuration parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `url` | Yes | Base URL of your API endpoint |
| `spec_path` | Yes | Path or URL to your OpenAPI spec file (JSON or YAML) |
| `auth_type` | No | `none`, `api_key`, `bearer_token`, `basic`, `authorization`, `oauth2` |
| `auth_value` | No | Auth value (required if `auth_type` is set) |
| `description` | No | Optional description for the MCP server |
| `allowed_tools` | No | List of specific tools to allow |
| `disallowed_tools` | No | List of specific tools to block |

**Supported OpenAPI versions:** 3.0.x, 3.1.x, Swagger 2.0

Each operation's `operationId` becomes the MCP tool name, so make sure your spec has unique `operationId` values.

## Using the Server

<Tabs>
<TabItem value="fastmcp" label="Python FastMCP">

```python title="Using OpenAPI-based MCP Server" showLineNumbers
from fastmcp import Client
import asyncio

config = {
    "mcpServers": {
        "petstore": {
            "url": "http://localhost:4000/petstore_mcp/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-1234"
            }
        }
    }
}

client = Client(config)

async def main():
    async with client:
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        response = await client.call_tool(
            name="getpetbyid",
            arguments={"petId": "1"}
        )
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

```json title="Cursor MCP Configuration" showLineNumbers
{
  "mcpServers": {
    "Petstore": {
      "url": "http://localhost:4000/petstore_mcp/mcp",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY"
      }
    }
  }
}
```

</TabItem>

<TabItem value="openai" label="OpenAI Responses API">

```bash title="Using OpenAPI MCP Server with OpenAI" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "petstore",
            "server_url": "http://localhost:4000/petstore_mcp/mcp",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
            }
        }
    ],
    "input": "Find all available pets in the petstore",
    "tool_choice": "required"
}'
```

</TabItem>
</Tabs>

## Overriding Tool Names and Descriptions

By default, tool names and descriptions come directly from the `operationId` and description fields in your OpenAPI spec. You can override these per-server so MCP clients see friendlier names and clearer descriptions — without touching the upstream spec.

This is useful when:
- The spec uses machine-generated `operationId` values like `getPetById_v2_deprecated`
- You want to simplify descriptions for your team
- You're exposing the same API to multiple audiences with different naming conventions

### Via the UI

When adding or editing an MCP server in the LiteLLM UI, each tool card in the **Tool Configuration** section has a pencil icon. Click it to open an inline edit panel:

- **Display Name** — overrides the tool name shown to MCP clients
- **Description** — overrides the tool description shown to MCP clients

A purple **Custom name** badge appears on any tool with an active override. Leave a field blank to keep the original value from the spec.

### Via the API

Pass `tool_name_to_display_name` and `tool_name_to_description` when creating or updating an MCP server:

```bash title="Create server with tool overrides" showLineNumbers
curl -X POST http://localhost:4000/v1/mcp/server \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "petstore_mcp",
    "url": "https://petstore.swagger.io/v2",
    "spec_path": "/path/to/openapi.json",
    "tool_name_to_display_name": {
      "getPetById": "Get Pet",
      "findPetsByStatus": "List Available Pets"
    },
    "tool_name_to_description": {
      "getPetById": "Look up a pet by its ID",
      "findPetsByStatus": "Returns all pets that match the given status (available, pending, sold)"
    }
  }'
```

```bash title="Update overrides on an existing server" showLineNumbers
curl -X PUT http://localhost:4000/v1/mcp/server/{server_id} \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name_to_display_name": {
      "getPetById": "Get Pet"
    },
    "tool_name_to_description": {
      "getPetById": "Look up a pet by its ID"
    }
  }'
```

### How It Works

The key used in both maps is the **original tool name** from the spec (the `operationId`), not the prefixed name. LiteLLM strips the server prefix before looking up overrides.

For example, if your server is named `petstore_mcp`, the tool is exposed as `petstore_mcp-getPetById`. The map key is still `getPetById`:

```json
{
  "tool_name_to_display_name": {
    "getPetById": "Get Pet"
  }
}
```

After the override, MCP clients will see `"Get Pet"` instead of `"petstore_mcp-getPetById"`.

### Example: Before and After

Admin configures on the `petstore_mcp` server:
```json
{
  "tool_name_to_display_name": {
    "getPetById": "Get Pet",
    "findPetsByStatus": "List Available Pets"
  },
  "tool_name_to_description": {
    "getPetById": "Look up a pet by its ID",
    "findPetsByStatus": "Returns all pets matching the given status"
  }
}
```

MCP client calls `tools/list` and sees:

```
Tool: "Get Pet"
Description: "Look up a pet by its ID"

Tool: "List Available Pets"
Description: "Returns all pets matching the given status"

Tool: "petstore_mcp-addPet"        ← no override, original name shown
Description: "Add a new pet to the store"
```
