# MCP Testing Guide for LiteLLM

## What are MCPs (Model Context Protocol)?

**Model Context Protocol (MCP)** is a standardized protocol that enables AI agents and LLMs to access external tools and services. Think of MCP servers as bridges that connect your LLM to real-world capabilities like:

- **GitHub operations** (create issues, search repositories)
- **Email services** (send emails via Gmail)
- **Productivity tools** (Zapier automations, Jira tickets)
- **Database access** (query and update data)
- **API integrations** (any REST API)

MCP servers expose **tools** that LLMs can discover and call. Each tool has a name, description, and input schema (parameters it accepts). When an LLM needs to perform an action, it can call these tools and receive structured results.

### Key Concepts

- **MCP Server**: A service that exposes tools via the MCP protocol
- **Tools**: Functions/operations that can be called by LLMs (e.g., `create_issue`, `send_email`)
- **Transports**: How MCP servers communicate:
  - **HTTP/SSE** (Server-Sent Events): Web-based communication
  - **STDIO** (Standard Input/Output): Process-based communication

## How LiteLLM Supports MCP

LiteLLM acts as an **MCP Gateway** that:

1. **Centralizes MCP server management**: Add MCP servers once in LiteLLM, use them across all LLM providers
2. **Provides unified access**: Use a single endpoint (`litellm_proxy`) to access all your MCP tools
3. **Manages permissions**: Control MCP access by API key, team, or organization
4. **Transforms protocols**: Converts between OpenAI tool format and MCP tool format automatically

### LiteLLM MCP Architecture

```
┌─────────────┐
│   Client    │  (Your application/agent)
└──────┬──────┘
       │ HTTP Requests
       ▼
┌─────────────────────────────┐
│   LiteLLM Proxy Gateway     │
│  - MCP Server Registry      │
│  - Permission Management    │
│  - Protocol Translation     │
└──────┬──────────────────────┘
       │ MCP Protocol
       ▼
┌─────────────┬─────────────┬─────────────┐
│  MCP Server │  MCP Server │  MCP Server │
│  (GitHub)   │  (Zapier)   │  (Custom)   │
└─────────────┴─────────────┴─────────────┘
```

### Supported Operations

- **List Tools**: Discover available tools from all registered MCP servers
- **Call Tools**: Execute tools with provided arguments
- **Permission Control**: Restrict tool access by API key, team, or organization

## Setting Up a REST Testing Environment

### Prerequisites

1. **LiteLLM Proxy running** with MCP support enabled
2. **At least one MCP server** configured (or a mock/test MCP server)
3. **API key** with MCP access permissions

### Step 1: Configure MCP Server in LiteLLM

Add your MCP server to `config.yaml`:

```yaml
general_settings:
  master_key: sk-1234
  store_model_in_db: true # Required for MCP servers in DB

mcp_servers:
  # Example: HTTP-based MCP server
  github_mcp:
    url: "https://api.githubcopilot.com/mcp"
    transport: "http"
    auth_type: "bearer_token"
    auth_value: "your-github-token"

  # Example: SSE-based MCP server
  zapier_mcp:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
    transport: "sse"

  # Example: STDIO-based MCP server
  local_mcp:
    transport: "stdio"
    command: "python3"
    args: ["./local_mcp_server.py"]
    env:
      API_KEY: "your-api-key"

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
```

**Note**: For testing purposes, you can use a simple HTTP mock server or public MCP servers like DeepWiki (`https://mcp.deepwiki.com/mcp`).

### Step 2: Start LiteLLM Proxy

```bash
litellm --config config.yaml
```

The proxy should be running on `http://localhost:4000` (default port).

### Step 3: REST API Endpoints for Testing

LiteLLM provides several REST endpoints for MCP operations:

#### 3.1 List All Available Tools

**Endpoint**: `GET /mcp-rest/tools/list`

**Authentication**: Requires LiteLLM API key in `Authorization` header

**Example Request**:

```bash
curl --location 'http://localhost:4000/mcp-rest/tools/list' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234'
```

**Query Parameters**:

- `server_id` (optional): Filter tools from a specific MCP server

**Example Response**:

```json
{
  "tools": [
    {
      "name": "github_create_issue",
      "description": "Create a new GitHub issue",
      "inputSchema": {
        "type": "object",
        "properties": {
          "title": { "type": "string" },
          "body": { "type": "string" }
        },
        "required": ["title"]
      },
      "mcp_info": {
        "server_name": "github_mcp",
        "logo_url": "https://github.com/logo.png"
      }
    }
  ],
  "error": null,
  "message": "Successfully retrieved tools"
}
```

#### 3.2 Call an MCP Tool

**Endpoint**: `POST /mcp-rest/tools/call`

**Authentication**: Requires LiteLLM API key in `Authorization` header

**Example Request**:

```bash
curl --location 'http://localhost:4000/mcp-rest/tools/call' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "tool_name": "github_create_issue",
    "arguments": {
        "title": "Test Issue",
        "body": "This is a test issue created via MCP"
    }
}'
```

**Request Body**:

```json
{
  "tool_name": "tool_name_from_list",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

**Example Response**:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Issue #123 created successfully"
    }
  ],
  "isError": false
}
```

#### 3.3 List Tools from Specific Server

**Endpoint**: `GET /mcp-rest/tools/list?server_id=<server_id>`

**Example**:

```bash
curl --location 'http://localhost:4000/mcp-rest/tools/list?server_id=github_mcp' \
--header 'Authorization: Bearer sk-1234'
```

### Step 4: Create a Load Testing Script (Similar to `no_cache_hits.py`)

Here's a Locust-based script to test MCP endpoints:

```python
import uuid
from locust import HttpUser, task, between, events

class MCPLoadTest(HttpUser):
    wait_time = between(0.5, 1)

    def on_start(self):
        """Set up authentication headers"""
        self.api_key = "sk-1234"
        self.client.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

    @task(3)
    def list_mcp_tools(self):
        """List all available MCP tools"""
        response = self.client.get(
            "/mcp-rest/tools/list",
            name="List MCP Tools"
        )

        if response.status_code == 200:
            data = response.json()
            # Log if we got tools
            if data.get("tools"):
                print(f"Found {len(data['tools'])} tools")

        if response.status_code != 200:
            with open("mcp_errors.txt", "a") as error_log:
                error_log.write(f"List tools error: {response.text}\n")

    @task(1)
    def call_mcp_tool(self):
        """Call an MCP tool (adjust tool_name and arguments based on your setup)"""
        # First, list tools to get a valid tool name
        list_response = self.client.get("/mcp-rest/tools/list")

        if list_response.status_code == 200:
            tools = list_response.json().get("tools", [])

            if tools:
                # Use the first available tool
                tool = tools[0]
                tool_name = tool["name"]

                # Create a minimal request with required parameters
                # Adjust based on your tool's schema
                payload = {
                    "tool_name": tool_name,
                    "arguments": {
                        # Add required arguments based on tool schema
                        "test_param": f"test-{uuid.uuid4()}"
                    }
                }

                response = self.client.post(
                    "/mcp-rest/tools/call",
                    json=payload,
                    name="Call MCP Tool"
                )

                if response.status_code != 200:
                    with open("mcp_errors.txt", "a") as error_log:
                        error_log.write(
                            f"Call tool error ({tool_name}): {response.text}\n"
                        )
```

**Run the load test**:

```bash
locust -f mcp_load_test.py --host=http://localhost:4000
```

### Step 5: Testing with OpenAI Responses API Format

For a more realistic test (similar to how clients actually use MCP), you can test using the Responses API format:

```python
import uuid
from locust import HttpUser, task, between

class MCPResponsesAPITest(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        self.api_key = "sk-1234"
        self.client.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

    @task
    def test_mcp_with_responses_api(self):
        """Test MCP tools via Responses API (more realistic)"""
        payload = {
            "model": "gpt-4o",  # Use your configured model
            "input": [
                {
                    "role": "user",
                    "content": f"Test MCP tool call {uuid.uuid4()}",
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
            "tool_choice": "required"
        }

        response = self.client.post(
            "/v1/responses",
            json=payload,
            name="MCP Responses API"
        )

        if response.status_code != 200:
            with open("mcp_errors.txt", "a") as error_log:
                error_log.write(f"Responses API error: {response.text}\n")
```

### Testing Tips

1. **Start Simple**: Begin with listing tools before calling them
2. **Check Tool Schemas**: Use the list endpoint to understand required parameters
3. **Monitor Errors**: Log errors to identify issues (connection failures, auth problems, etc.)
4. **Test Incrementally**:
   - First test with one MCP server
   - Then test with multiple servers
   - Finally test under load
5. **Mock Servers**: Consider using mock MCP servers for load testing to avoid rate limits

### Common Issues and Solutions

1. **No tools returned**:

   - Verify MCP server is configured correctly in `config.yaml`
   - Check MCP server is accessible from LiteLLM proxy
   - Verify authentication credentials are correct

2. **Connection errors**:

   - Ensure MCP server URL is correct
   - Check network connectivity
   - Verify transport type matches server capabilities

3. **Permission errors**:
   - Check API key has access to MCP servers
   - Verify team/organization permissions if using access groups

### Additional Resources

- **LiteLLM MCP Documentation**: https://docs.litellm.ai/docs/mcp
- **MCP Protocol Specification**: https://modelcontextprotocol.io
- **LiteLLM Proxy Configuration**: https://docs.litellm.ai/docs/proxy/config_settings
