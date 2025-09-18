import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# /mcp - Model Context Protocol

LiteLLM Proxy provides an MCP Gateway that allows you to use a fixed endpoint for all MCP tools and control MCP access by Key, Team. 

<Image 
  img={require('../img/mcp_2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  LiteLLM MCP Architecture: Use MCP tools with all LiteLLM supported models
</p>

## Overview
| Feature | Description |
|---------|-------------|
| MCP Operations | • List Tools<br/>• Call Tools |
| Supported MCP Transports | • Streamable HTTP<br/>• SSE<br/>• Standard Input/Output (stdio) |
| LiteLLM Permission Management | • By Key<br/>• By Team<br/>• By Organization |

## Adding your MCP

<Tabs>
<TabItem value="ui" label="LiteLLM UI">

On the LiteLLM UI, Navigate to "MCP Servers" and click "Add New MCP Server".

On this form, you should enter your MCP Server URL and the transport you want to use.

LiteLLM supports the following MCP transports:
- Streamable HTTP
- SSE (Server-Sent Events)
- Standard Input/Output (stdio)

<Image 
  img={require('../img/add_mcp.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>
<br/>

### Add HTTP MCP Server

This video walks through adding and using an HTTP MCP server on LiteLLM UI and using it in Cursor IDE.

<iframe width="840" height="500" src="https://www.loom.com/embed/e2aebce78e8d46beafeb4bacdde31f14" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

<br/>
<br/>

### Add SSE MCP Server

This video walks through adding and using an SSE MCP server on LiteLLM UI and using it in Cursor IDE.

<iframe width="840" height="500" src="https://www.loom.com/embed/07e04e27f5e74475b9cf8ef8247d2c3e" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

<br/>
<br/>

### Add STDIO MCP Server

For stdio MCP servers, select "Standard Input/Output (stdio)" as the transport type and provide the stdio configuration in JSON format:

<Image 
  img={require('../img/add_stdio_mcp.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

</TabItem>

<TabItem value="config" label="config.yaml">

Add your MCP servers directly in your `config.yaml` file:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

litellm_settings:
  # MCP Aliases - Map aliases to server names for easier tool access
  mcp_aliases:
    "github": "github_mcp_server"
    "zapier": "zapier_mcp_server"
    "deepwiki": "deepwiki_mcp_server"

mcp_servers:
  # HTTP Streamable Server
  deepwiki_mcp:
    url: "https://mcp.deepwiki.com/mcp"
  # SSE Server
  zapier_mcp:
    url: "https://actions.zapier.com/mcp/sk-akxxxxx/sse"
  
  # Standard Input/Output (stdio) Server - CircleCI Example
  circleci_mcp:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@circleci/mcp-server-circleci"]
    env:
      CIRCLECI_TOKEN: "your-circleci-token"
      CIRCLECI_BASE_URL: "https://circleci.com"
  
  # Full configuration with all optional fields
  my_http_server:
    url: "https://my-mcp-server.com/mcp"
    transport: "http"
    description: "My custom MCP server"
    auth_type: "api_key"
    auth_value: "abc123"
```

**Configuration Options:**
- **Server Name**: Use any descriptive name for your MCP server (e.g., `zapier_mcp`, `deepwiki_mcp`, `circleci_mcp`)
- **Alias**: This name will be prefilled with the server name with "_" replacing spaces, else edit it to be the prefix in tool names
- **URL**: The endpoint URL for your MCP server (required for HTTP/SSE transports)
- **Transport**: Optional transport type (defaults to `sse`)
  - `sse` - SSE (Server-Sent Events) transport
  - `http` - Streamable HTTP transport
  - `stdio` - Standard Input/Output transport
- **Command**: The command to execute for stdio transport (required for stdio)
- **Args**: Array of arguments to pass to the command (optional for stdio)
- **Env**: Environment variables to set for the stdio process (optional for stdio)
- **Description**: Optional description for the server
- **Auth Type**: Optional authentication type. Supported values:

  | Value | Header sent |
  |-------|-------------|
  | `api_key` | `X-API-Key: <auth_value>` |
  | `bearer_token` | `Authorization: Bearer <auth_value>` |
  | `basic` | `Authorization: Basic <auth_value>` |
  | `authorization` | `Authorization: <auth_value>` |

- **Spec Version**: Optional MCP specification version (defaults to `2025-06-18`)

Examples for each auth type:

```yaml title="MCP auth examples (config.yaml)" showLineNumbers
mcp_servers:
  api_key_example:
    url: "https://my-mcp-server.com/mcp"
    auth_type: "api_key"
    auth_value: "abc123"        # headers={"X-API-Key": "abc123"}

  bearer_example:
    url: "https://my-mcp-server.com/mcp"
    auth_type: "bearer_token"
    auth_value: "abc123"        # headers={"Authorization": "Bearer abc123"}

  basic_example:
    url: "https://my-mcp-server.com/mcp"
    auth_type: "basic"
    auth_value: "dXNlcjpwYXNz"  # headers={"Authorization": "Basic dXNlcjpwYXNz"}

  custom_auth_example:
    url: "https://my-mcp-server.com/mcp"
    auth_type: "authorization"
    auth_value: "Token example123"  # headers={"Authorization": "Token example123"}
```


### MCP Aliases

You can define aliases for your MCP servers in the `litellm_settings` section. This allows you to:

1. **Map friendly names to server names**: Use shorter, more memorable aliases
2. **Override server aliases**: If a server doesn't have an alias defined, the system will use the first matching alias from `mcp_aliases`
3. **Ensure uniqueness**: Only the first alias for each server is used, preventing conflicts

**Example:**
```yaml
litellm_settings:
  mcp_aliases:
    "github": "github_mcp_server"      # Maps "github" alias to "github_mcp_server"
    "zapier": "zapier_mcp_server"      # Maps "zapier" alias to "zapier_mcp_server"
    "docs": "deepwiki_mcp_server"      # Maps "docs" alias to "deepwiki_mcp_server"
    "github_alt": "github_mcp_server"  # This will be ignored since "github" already maps to this server
```

**Benefits:**
- **Simplified tool access**: Use `github_create_issue` instead of `github_mcp_server_create_issue`
- **Consistent naming**: Standardize alias patterns across your organization
- **Easy migration**: Change server names without breaking existing tool references

</TabItem>
</Tabs>


## Using your MCP

### Use on LiteLLM UI 

Follow this walkthrough to use your MCP on LiteLLM UI

<iframe width="840" height="500" src="https://www.loom.com/embed/57e0763267254bc79dbe6658d0b8758c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### Use with Responses API

Replace `http://localhost:4000` with your LiteLLM Proxy base URL.

Demo Video Using Responses API with LiteLLM Proxy: [Demo video here](https://www.loom.com/share/34587e618c5c47c0b0d67b4e4d02718f?sid=2caf3d45-ead4-4490-bcc1-8d6dd6041c02)


<Tabs>
<TabItem value="curl" label="cURL">

```bash title="cURL Example" showLineNumbers
curl --location 'http://localhost:4000/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer sk-1234" \
--data '{
    "model": "gpt-5",
    "input": [
    {
      "role": "user",
      "content": "give me TLDR of what BerriAI/litellm repo is about",
      "type": "message"
    }
  ],
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        }
    ],
    "stream": true,
    "tool_choice": "required"
}'
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python title="Python SDK Example" showLineNumbers
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
```

</TabItem>
</Tabs>

#### Specifying MCP Tools

You can specify which MCP tools are available by using the `allowed_tools` parameter. This allows you to restrict access to specific tools within an MCP server.

To get the list of allowed tools when using LiteLLM MCP Gateway, you can naigate to the LiteLLM UI on MCP Servers > MCP Tools > Click the Tool > Copy Tool Name.

<Tabs>
<TabItem value="curl" label="cURL">

```bash title="cURL Example with allowed_tools" showLineNumbers
curl --location 'http://localhost:4000/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer sk-1234" \
--data '{
    "model": "gpt-5",
    "input": [
    {
      "role": "user",
      "content": "give me TLDR of what BerriAI/litellm repo is about",
      "type": "message"
    }
  ],
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["GitMCP-fetch_litellm_documentation"]
        }
    ],
    "stream": true,
    "tool_choice": "required"
}'
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python title="Python SDK Example with allowed_tools" showLineNumbers
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

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
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["GitMCP-fetch_litellm_documentation"]
        }
    ],
    stream=True,
    tool_choice="required"
)

print(response)
```

</TabItem>
</Tabs>

### Use with Cursor IDE

Use tools directly from Cursor IDE with LiteLLM MCP:

**Setup Instructions:**

1. **Open Cursor Settings**: Use `⇧+⌘+J` (Mac) or `Ctrl+Shift+J` (Windows/Linux)
2. **Navigate to MCP Tools**: Go to the "MCP Tools" tab and click "New MCP Server"
3. **Add Configuration**: Copy and paste the JSON configuration below, then save with `Cmd+S` or `Ctrl+S`

```json title="Basic Cursor MCP Configuration" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY"
      }
    }
  }
}
```

#### How it works when server_url="litellm_proxy"

When server_url="litellm_proxy", LiteLLM bridges non-MCP providers to your MCP tools.

- Tool Discovery: LiteLLM fetches MCP tools and converts them to OpenAI-compatible definitions
- LLM Call: Tools are sent to the LLM with your input; LLM selects which tools to call
- Tool Execution: LiteLLM automatically parses arguments, routes calls to MCP servers, executes tools, and retrieves results
- Response Integration: Tool results are sent back to LLM for final response generation
- Output: Complete response combining LLM reasoning with tool execution results

This enables MCP tool usage with any LiteLLM-supported provider, regardless of native MCP support.

#### Auto-execution for require_approval: "never"

Setting require_approval: "never" triggers automatic tool execution, returning the final response in a single API call without additional user interaction.



## MCP Server Access Control

LiteLLM Proxy provides two methods for controlling access to specific MCP servers:

1. **URL-based Namespacing** - Use URL paths to directly access specific servers or access groups
2. **Header-based Namespacing** - Use the `x-mcp-servers` header to specify which servers to access

---

### Method 1: URL-based Namespacing

LiteLLM Proxy supports URL-based namespacing for MCP servers using the format `/mcp/<servers or access groups>`. This allows you to:

- **Direct URL Access**: Point MCP clients directly to specific servers or access groups via URL
- **Simplified Configuration**: Use URLs instead of headers for server selection
- **Access Group Support**: Use access group names in URLs for grouped server access

#### URL Format

```
<your-litellm-proxy-base-url>/mcp/<server_alias_or_access_group>
```

**Examples:**
- `/mcp/github` - Access tools from the "github" MCP server
- `/mcp/zapier` - Access tools from the "zapier" MCP server  
- `/mcp/dev_group` - Access tools from all servers in the "dev_group" access group
- `/mcp/github,zapier` - Access tools from multiple specific servers

#### Usage Examples

<Tabs>
<TabItem value="openai" label="OpenAI API">

```bash title="cURL Example with URL Namespacing" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "<your-litellm-proxy-base-url>/mcp/github",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

This example uses URL namespacing to access only the "github" MCP server.

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

```bash title="cURL Example with URL Namespacing" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "<your-litellm-proxy-base-url>/mcp/dev_group",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

This example uses URL namespacing to access all servers in the "dev_group" access group.

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

```json title="Cursor MCP Configuration with URL Namespacing" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "<your-litellm-proxy-base-url>/mcp/github,zapier",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY"
      }
    }
  }
}
```

This configuration uses URL namespacing to access tools from both "github" and "zapier" MCP servers.

</TabItem>
</Tabs>

#### Benefits of URL Namespacing

- **Direct Access**: No need for additional headers to specify servers
- **Clean URLs**: Self-documenting URLs that clearly indicate which servers are accessible
- **Access Group Support**: Use access group names for grouped server access
- **Multiple Servers**: Specify multiple servers in a single URL with comma separation
- **Simplified Configuration**: Easier setup for MCP clients that prefer URL-based configuration

---

### Method 2: Header-based Namespacing

You can choose to access specific MCP servers and only list their tools using the `x-mcp-servers` header. This header allows you to:
- Limit tool access to one or more specific MCP servers
- Control which tools are available in different environments or use cases

The header accepts a comma-separated list of server aliases: `"alias_1,Server2,Server3"`

**Notes:**
- If the header is not provided, tools from all available MCP servers will be accessible
- This method works with the standard LiteLLM MCP endpoint

<Tabs>
<TabItem value="openai" label="OpenAI API">

```bash title="cURL Example with Header Namespacing" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "<your-litellm-proxy-base-url>/mcp/",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "alias_1"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

In this example, the request will only have access to tools from the "alias_1" MCP server.

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

```bash title="cURL Example with Header Namespacing" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "<your-litellm-proxy-base-url>/mcp/",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "alias_1,Server2"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

This configuration restricts the request to only use tools from the specified MCP servers.

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

```json title="Cursor MCP Configuration with Header Namespacing" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "<your-litellm-proxy-base-url>/mcp/",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-servers": "alias_1,Server2"
      }
    }
  }
}
```

This configuration in Cursor IDE settings will limit tool access to only the specified MCP servers.

</TabItem>
</Tabs>

---

### Comparison: Header vs URL Namespacing

| Feature | Header Namespacing | URL Namespacing |
|---------|-------------------|-----------------|
| **Method** | Uses `x-mcp-servers` header | Uses URL path `/mcp/<servers>` |
| **Endpoint** | Standard `litellm_proxy` endpoint | Custom `/mcp/<servers>` endpoint |
| **Configuration** | Requires additional header | Self-contained in URL |
| **Multiple Servers** | Comma-separated in header | Comma-separated in URL path |
| **Access Groups** | Supported via header | Supported via URL path |
| **Client Support** | Works with all MCP clients | Works with URL-aware MCP clients |
| **Use Case** | Dynamic server selection | Fixed server configuration |

<Tabs>
<TabItem value="openai" label="OpenAI API">

```bash title="cURL Example with Server Segregation" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "<your-litellm-proxy-base-url>/mcp/",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "alias_1"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

In this example, the request will only have access to tools from the "alias_1" MCP server.

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

```bash title="cURL Example with Server Segregation" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "alias_1,Server2"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

This configuration restricts the request to only use tools from the specified MCP servers.

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

```json title="Cursor MCP Configuration with Server Segregation" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-servers": "alias_1,Server2"
      }
    }
  }
}
```

This configuration in Cursor IDE settings will limit tool access to only the specified MCP server.

</TabItem>
</Tabs>

### Grouping MCPs (Access Groups)

MCP Access Groups allow you to group multiple MCP servers together for easier management.

#### 1. Create an Access Group

##### A. Creating Access Groups using Config:

```yaml title="Creating access groups for MCP using the config" showLineNumbers
mcp_servers:
  "deepwiki_mcp":
    url: https://mcp.deepwiki.com/mcp
    transport: "http"
    auth_type: "none"
    access_groups: ["dev_group"]
```

While adding `mcp_servers` using the config:
- Pass in a list of strings inside `access_groups`
- These groups can then be used for segregating access using keys, teams and MCP clients using headers

##### B. Creating Access Groups using UI

To create an access group:
- Go to MCP Servers in the LiteLLM UI
- Click "Add a New MCP Server" 
- Under "MCP Access Groups", create a new group (e.g., "dev_group") by typing it
- Add the same group name to other servers to group them together

<Image 
  img={require('../img/mcp_create_access_group.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

#### 2. Use Access Group in Cursor

Include the access group name in the `x-mcp-servers` header:

```json title="Cursor Configuration with Access Groups" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-servers": "dev_group"
      }
    }
  }
}
```

This gives you access to all servers in the "dev_group" access group.
- Which means that if deepwiki server (and any other servers) which have the access group `dev_group` assigned to them will be available for tool calling

#### Advanced: Connecting Access Groups to API Keys

When creating API keys, you can assign them to specific access groups for permission management:

- Go to "Keys" in the LiteLLM UI and click "Create Key"
- Select the desired MCP access groups from the dropdown
- The key will have access to all MCP servers in those groups
- This is reflected in the Test Key page

<Image 
  img={require('../img/mcp_key_access_group.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>


## Using your MCP with client side credentials

Use this if you want to pass a client side authentication token to LiteLLM to then pass to your MCP to auth to your MCP.


### New Server-Specific Auth Headers (Recommended)

You can specify MCP auth tokens using server-specific headers in the format `x-mcp-{server_alias}-{header_name}`. This allows you to use different authentication for different MCP servers.

**Format:** `x-mcp-{server_alias}-{header_name}: value`

**Examples:**
- `x-mcp-github-authorization: Bearer ghp_xxxxxxxxx` - GitHub MCP server with Bearer token
- `x-mcp-zapier-x-api-key: sk-xxxxxxxxx` - Zapier MCP server with API key
- `x-mcp-deepwiki-authorization: Basic base64_encoded_creds` - DeepWiki MCP server with Basic auth

**Benefits:**
- **Server-specific authentication**: Each MCP server can use different auth methods
- **Better security**: No need to share the same auth token across all servers
- **Flexible header names**: Support for different auth header types (authorization, x-api-key, etc.)
- **Clean separation**: Each server's auth is clearly identified

### Legacy Auth Header (Deprecated)

You can also specify your MCP auth token using the header `x-mcp-auth`. This will be forwarded to all MCP servers and is deprecated in favor of server-specific headers.

<Tabs>
<TabItem value="openai" label="OpenAI API">

#### Connect via OpenAI Responses API with Server-Specific Auth

Use the OpenAI Responses API and include server-specific auth headers:

```bash title="cURL Example with Server-Specific Auth" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-github-authorization": "Bearer YOUR_GITHUB_TOKEN",
                "x-mcp-zapier-x-api-key": "YOUR_ZAPIER_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

#### Connect via OpenAI Responses API with Legacy Auth

Use the OpenAI Responses API and include the `x-mcp-auth` header for your MCP server authentication:

```bash title="cURL Example with Legacy MCP Auth" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-auth": YOUR_MCP_AUTH_TOKEN
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

#### Connect via LiteLLM Proxy Responses API with Server-Specific Auth

Use this when calling LiteLLM Proxy for LLM API requests to `/v1/responses` endpoint with server-specific authentication:

```bash title="cURL Example with Server-Specific Auth" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-github-authorization": "Bearer YOUR_GITHUB_TOKEN",
                "x-mcp-zapier-x-api-key": "YOUR_ZAPIER_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

#### Connect via LiteLLM Proxy Responses API with Legacy Auth

Use this when calling LiteLLM Proxy for LLM API requests to `/v1/responses` endpoint with MCP authentication:

```bash title="cURL Example with Legacy MCP Auth" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-auth": "YOUR_MCP_AUTH_TOKEN"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

#### Connect via Cursor IDE with Server-Specific Auth

Use tools directly from Cursor IDE with LiteLLM MCP and include server-specific authentication:

**Setup Instructions:**

1. **Open Cursor Settings**: Use `⇧+⌘+J` (Mac) or `Ctrl+Shift+J` (Windows/Linux)
2. **Navigate to MCP Tools**: Go to the "MCP Tools" tab and click "New MCP Server"
3. **Add Configuration**: Copy and paste the JSON configuration below, then save with `Cmd+S` or `Ctrl+S`

```json title="Cursor MCP Configuration with Server-Specific Auth" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-github-authorization": "Bearer $GITHUB_TOKEN",
        "x-mcp-zapier-x-api-key": "$ZAPIER_API_KEY"
      }
    }
  }
}
```

#### Connect via Cursor IDE with Legacy Auth

Use tools directly from Cursor IDE with LiteLLM MCP and include your MCP authentication token:

**Setup Instructions:**

1. **Open Cursor Settings**: Use `⇧+⌘+J` (Mac) or `Ctrl+Shift+J` (Windows/Linux)
2. **Navigate to MCP Tools**: Go to the "MCP Tools" tab and click "New MCP Server"
3. **Add Configuration**: Copy and paste the JSON configuration below, then save with `Cmd+S` or `Ctrl+S`

```json title="Cursor MCP Configuration with Legacy Auth" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-auth": "$MCP_AUTH_TOKEN"
      }
    }
  }
}
```

</TabItem>

<TabItem value="http" label="Streamable HTTP">

#### Connect via Streamable HTTP Transport with Server-Specific Auth

Connect to LiteLLM MCP using HTTP transport with server-specific authentication:

**Server URL:**
```text showLineNumbers
litellm_proxy
```

**Headers:**
```text showLineNumbers
x-litellm-api-key: Bearer YOUR_LITELLM_API_KEY
x-mcp-github-authorization: Bearer YOUR_GITHUB_TOKEN
x-mcp-zapier-x-api-key: YOUR_ZAPIER_API_KEY
```

#### Connect via Streamable HTTP Transport with Legacy Auth

Connect to LiteLLM MCP using HTTP transport with MCP authentication:

**Server URL:**
```text showLineNumbers
litellm_proxy
```

**Headers:**
```text showLineNumbers
x-litellm-api-key: Bearer YOUR_LITELLM_API_KEY
x-mcp-auth: Bearer YOUR_MCP_AUTH_TOKEN
```

This URL can be used with any MCP client that supports HTTP transport. The `x-mcp-auth` header will be forwarded to your MCP server for authentication.

</TabItem>

<TabItem value="fastmcp" label="Python FastMCP">

#### Connect via Python FastMCP Client with Server-Specific Auth

Use the Python FastMCP client to connect to your LiteLLM MCP server with server-specific authentication:

```python title="Python FastMCP Example with Server-Specific Auth" showLineNumbers
import asyncio
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Create the transport with your LiteLLM MCP server URL and server-specific auth headers
server_url = "litellm_proxy"
transport = StreamableHttpTransport(
    server_url,
    headers={
        "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
        "x-mcp-github-authorization": "Bearer YOUR_GITHUB_TOKEN",
        "x-mcp-zapier-x-api-key": "YOUR_ZAPIER_API_KEY"
    }
)

# Initialize the client with the transport
client = Client(transport=transport)


async def main():
    # Connection is established here
    print("Connecting to LiteLLM MCP server with server-specific authentication...")
    async with client:
        print(f"Client connected: {client.is_connected()}")

        # Make MCP calls within the context
        print("Fetching available tools...")
        tools = await client.list_tools()

        print(f"Available tools: {json.dumps([t.name for t in tools], indent=2)}")
        
        # Example: Call a tool (replace 'tool_name' with an actual tool name)
        if tools:
            tool_name = tools[0].name
            print(f"Calling tool: {tool_name}")
            
            # Call the tool with appropriate arguments
            result = await client.call_tool(tool_name, arguments={})
            print(f"Tool result: {result}")


# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

#### Connect via Python FastMCP Client with Legacy Auth

Use the Python FastMCP client to connect to your LiteLLM MCP server with MCP authentication:

```python title="Python FastMCP Example with Legacy MCP Auth" showLineNumbers
import asyncio
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Create the transport with your LiteLLM MCP server URL and auth headers
server_url = "litellm_proxy"
transport = StreamableHttpTransport(
    server_url,
    headers={
        "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
        "x-mcp-auth": "Bearer YOUR_MCP_AUTH_TOKEN"
    }
)

# Initialize the client with the transport
client = Client(transport=transport)


async def main():
    # Connection is established here
    print("Connecting to LiteLLM MCP server with authentication...")
    async with client:
        print(f"Client connected: {client.is_connected()}")

        # Make MCP calls within the context
        print("Fetching available tools...")
        tools = await client.list_tools()

        print(f"Available tools: {json.dumps([t.name for t in tools], indent=2)}")
        
        # Example: Call a tool (replace 'tool_name' with an actual tool name)
        if tools:
            tool_name = tools[0].name
            print(f"Calling tool: {tool_name}")
            
            # Call the tool with appropriate arguments
            result = await client.call_tool(tool_name, arguments={})
            print(f"Tool result: {result}")


# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>
</Tabs>

### Customize the MCP Auth Header Name

By default, LiteLLM uses `x-mcp-auth` to pass your credentials to MCP servers. You can change this header name in one of the following ways:
1. Set the `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` environment variable

```bash title="Environment Variable" showLineNumbers
export LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME="authorization"
```


2. Set the `mcp_client_side_auth_header_name` in the general settings on the config.yaml file

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

general_settings:
  mcp_client_side_auth_header_name: "authorization"
```

#### Using the authorization header

In this example the `authorization` header will be passed to the MCP server for authentication.

```bash title="cURL with authorization header" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "authorization": "Bearer sk-zapier-token-123"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```



## MCP Cost Tracking

LiteLLM provides two ways to track costs for MCP tool calls:

| Method | When to Use | What It Does |
|--------|-------------|--------------|
| **Config-based Cost Tracking** | Simple cost tracking with fixed costs per tool/server | Automatically tracks costs based on configuration |
| **Custom Post-MCP Hook** | Dynamic cost tracking with custom logic | Allows custom cost calculations and response modifications |

### Config-based Cost Tracking

Configure fixed costs for MCP servers directly in your config.yaml:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

mcp_servers:
  zapier_server:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
    mcp_info:
      mcp_server_cost_info:
        # Default cost for all tools in this server
        default_cost_per_query: 0.01
        # Custom cost for specific tools
        tool_name_to_cost_per_query:
          send_email: 0.05
          create_document: 0.03
          
  expensive_api_server:
    url: "https://api.expensive-service.com/mcp"
    mcp_info:
      mcp_server_cost_info:
        default_cost_per_query: 1.50
```

### Custom Post-MCP Hook

Use this when you need dynamic cost calculation or want to modify the MCP response before it's returned to the user.

#### 1. Create a custom MCP hook file

```python title="custom_mcp_hook.py" showLineNumbers
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.mcp import MCPPostCallResponseObject


class CustomMCPCostTracker(CustomLogger):
    """
    Custom handler for MCP cost tracking and response modification
    """
    
    async def async_post_mcp_tool_call_hook(
        self, 
        kwargs, 
        response_obj: MCPPostCallResponseObject, 
        start_time, 
        end_time
    ) -> Optional[MCPPostCallResponseObject]:
        """
        Called after each MCP tool call. 
        Modify costs and response before returning to user.
        """
        
        # Extract tool information from kwargs
        tool_name = kwargs.get("name", "")
        server_name = kwargs.get("server_name", "")
        
        # Calculate custom cost based on your logic
        custom_cost = 42.00
        
        # Set the response cost
        response_obj.hidden_params.response_cost = custom_cost
        
  
      
        return response_obj
    

# Create instance for LiteLLM to use
custom_mcp_cost_tracker = CustomMCPCostTracker()
```

#### 2. Configure in config.yaml

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

# Add your custom MCP hook
callbacks:
  - custom_mcp_hook.custom_mcp_cost_tracker

mcp_servers:
  zapier_server:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
```

#### 3. Start the proxy

```shell
$ litellm --config /path/to/config.yaml 
```

When MCP tools are called, your custom hook will:
1. Calculate costs based on your custom logic
2. Modify the response if needed
3. Track costs in LiteLLM's logging system

## MCP Guardrails

LiteLLM supports applying guardrails to MCP tool calls to ensure security and compliance. You can configure guardrails to run before or during MCP calls to validate inputs and block or mask sensitive information.

### Supported MCP Guardrail Modes

MCP guardrails support the following modes:

- `pre_mcp_call`: Run **before** MCP call, on **input**. Use this mode when you want to apply validation/masking/blocking for MCP requests
- `during_mcp_call`: Run **during** MCP call execution. Use this mode for real-time monitoring and intervention

### Configuration Examples

Configure guardrails to run before MCP tool calls to validate and sanitize inputs:

```yaml title="config.yaml" showLineNumbers
guardrails:
  - guardrail_name: "mcp-input-validation"
    litellm_params:
      guardrail: presidio  # or other supported guardrails
      mode: "pre_mcp_call" # or during_mcp_call
      pii_entities_config:
        CREDIT_CARD: "BLOCK"  # Will block requests containing credit card numbers
        EMAIL_ADDRESS: "MASK"  # Will mask email addresses
        PHONE_NUMBER: "MASK"   # Will mask phone numbers
      default_on: true
```


### Usage Examples

#### Testing Pre-MCP Call Guardrails

Test your MCP guardrails with a request that includes sensitive information:

```bash title="Test MCP Guardrail" showLineNumbers
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is john@example.com"}
    ],
    "guardrails": ["mcp-input-validation"]
  }'
```

The request will be processed as follows:
1. Credit card number will be blocked (request rejected)
2. Email address will be masked (e.g., replaced with `<EMAIL_ADDRESS>`)

#### Using with MCP Tools

When using MCP tools, guardrails will be applied to the tool inputs:

```python title="Python Example with MCP Guardrails" showLineNumbers
import openai

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:4000"
)

# This request will trigger MCP guardrails
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Send an email to 555-123-4567 with my SSN 123-45-6789"}
    ],
    tools=[{"type": "mcp", "server_label": "litellm", "server_url": "litellm_proxy"}],
    guardrails=["mcp-input-validation"]
)
```

### Supported Guardrail Providers

MCP guardrails work with all LiteLLM-supported guardrail providers:

- **Presidio**: PII detection and masking
- **Bedrock**: AWS Bedrock guardrails
- **Lakera**: Content moderation
- **Aporia**: Custom guardrails
- **Custom**: Your own guardrail implementations

## MCP Permission Management

LiteLLM supports managing permissions for MCP Servers by Keys, Teams, Organizations (entities) on LiteLLM. When a MCP client attempts to list tools, LiteLLM will only return the tools the entity has permissions to access.

When Creating a Key, Team, or Organization, you can select the allowed MCP Servers that the entity has access to.

<Image 
  img={require('../img/mcp_key.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>


## LiteLLM Proxy - Walk through MCP Gateway
LiteLLM exposes an MCP Gateway for admins to add all their MCP servers to LiteLLM. The key benefits of using LiteLLM Proxy with MCP are:

1. Use a fixed endpoint for all MCP tools
2. MCP Permission management by Key, Team, or User

This video demonstrates how you can onboard an MCP server to LiteLLM Proxy, use it and set access controls.

<iframe width="840" height="500" src="https://www.loom.com/embed/f7aa8d217879430987f3e64291757bfc" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

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