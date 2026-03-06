import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# MCP from OpenAPI Specs

LiteLLM can convert any OpenAPI/Swagger spec into an MCP server — no custom MCP server code required.

## Step 1 — Add the MCP Server

Add your OpenAPI-based server in `config.yaml`:

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  petstore_mcp:
    url: "https://petstore.swagger.io/v2"
    spec_path: "/path/to/openapi.json"
    auth_type: "none"

  my_api_mcp:
    url: "http://0.0.0.0:8090"
    spec_path: "/path/to/openapi.json"
    auth_type: "api_key"
    auth_value: "your-api-key-here"

  secured_api_mcp:
    url: "https://api.example.com"
    spec_path: "/path/to/openapi.json"
    auth_type: "bearer_token"
    auth_value: "your-bearer-token"
```

Or from the UI: go to **MCP Servers → Add New MCP Server**, fill in the URL and spec path, and LiteLLM will fetch the spec and load all endpoints as tools.

**Configuration parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `url` | Yes | Base URL of your API |
| `spec_path` | Yes | Path or URL to your OpenAPI spec (JSON or YAML) |
| `auth_type` | No | `none`, `api_key`, `bearer_token`, `basic`, `authorization`, `oauth2` |
| `auth_value` | No | Auth value (required if `auth_type` is set) |
| `description` | No | Optional description |
| `allowed_tools` | No | Allowlist of specific tools |
| `disallowed_tools` | No | Blocklist of specific tools |

**Supported spec versions:** OpenAPI 3.0.x, 3.1.x, Swagger 2.0. Each operation's `operationId` becomes the tool name — make sure they're unique.

Once tools are loaded, you'll see them in the Tool Configuration section:

<Image
  img={require('../img/mcp_openapi_tools_loaded.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>

## Step 2 — Optionally Override Tool Names and Descriptions

By default, tool names and descriptions come from the `operationId` and description fields in your spec. You can rename or rewrite them so MCP clients see something cleaner — without touching the upstream spec.

### From the UI

Each tool card has a pencil icon. Click it to open the inline editor:

<Image
  img={require('../img/mcp_openapi_tool_edit_panel.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>

- **Display Name** — overrides the name MCP clients see
- **Description** — overrides the description MCP clients see
- Leave a field blank to keep the original from the spec

After setting overrides, a purple **Custom name** badge appears on the tool card:

<Image
  img={require('../img/mcp_openapi_custom_name_badge.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>

### From the API

Pass `tool_name_to_display_name` and `tool_name_to_description` in the create or update request:

```bash title="Create server with tool name overrides" showLineNumbers
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
      "findPetsByStatus": "Returns all pets matching a given status (available, pending, sold)"
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

The map key is the **original `operationId`** from the spec — not the prefixed tool name. LiteLLM strips the server prefix before doing the lookup.

For example, if your server is `petstore_mcp`, the tool is exposed as `petstore_mcp-getPetById`. The map key is still `getPetById`.

**Before and after:**

```
# Without overrides
Tool: "petstore_mcp-getPetById"
Description: "Returns a single pet"

Tool: "petstore_mcp-findPetsByStatus"
Description: "Finds Pets by status"

# After overrides
Tool: "Get Pet"
Description: "Look up a pet by its ID"

Tool: "List Available Pets"
Description: "Returns all pets matching a given status (available, pending, sold)"
```

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
            name="Get Pet",        # overridden name
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
    "input": "Find all available pets",
    "tool_choice": "required"
}'
```

</TabItem>
</Tabs>
