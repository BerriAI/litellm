# Stdio MCP Support Implementation

This document summarizes the implementation of stdio MCP support in LiteLLM.

## Overview

Added support for stdio (Standard Input/Output) transport in MCP (Model Context Protocol) servers, allowing LiteLLM to connect to MCP servers that run as command-line processes.

## Changes Made

### 1. Backend Changes

#### Type Definitions (`litellm/types/mcp.py`)
- Added `stdio` to `MCPTransport` enum
- Updated `MCPTransportType` literal to include `stdio`
- Added `MCPStdioConfig` TypedDict with fields:
  - `command`: str - Command to run (e.g., 'npx', 'python', 'node')
  - `args`: List[str] - Arguments to pass to the command
  - `env`: Optional[Dict[str, str]] - Environment variables

#### MCP Client (`litellm/experimental_mcp_client/client.py`)
- Added imports for `StdioServerParameters` and `stdio_client`
- Updated `MCPClient.__init__()` to accept `stdio_config` parameter
- Modified `connect()` method to handle stdio transport:
  - Creates `StdioServerParameters` with command, args, and env
  - Uses `stdio_client()` for connection
- Made `server_url` optional (not needed for stdio)

#### Proxy Types (`litellm/proxy/_types.py`)
- Updated `NewMCPServerRequest` to include stdio fields:
  - `command`: Optional[str]
  - `args`: Optional[List[str]]
  - `env`: Optional[Dict[str, str]]
- Added validation to ensure required fields are present based on transport type
- Made `url` optional (not required for stdio)
- Updated `UpdateMCPServerRequest` and `LiteLLM_MCPServerTable` similarly

#### MCP Server Manager (`litellm/proxy/_experimental/mcp_server/mcp_server_manager.py`)
- Updated `_create_mcp_client()` to handle stdio configuration
- Extracts stdio config from server object when transport is stdio

#### Server Type (`litellm/types/mcp_server/mcp_server_manager.py`)
- Added stdio fields to `MCPServer` BaseModel
- Made `url` optional to support stdio servers

### 2. Database Changes

#### Migration (`litellm-proxy-extras/litellm_proxy_extras/migrations/20250125000000_add_stdio_fields_to_mcp_table/migration.sql`)
- Added `command`, `args`, and `env` columns to `LiteLLM_MCPServerTable`
- Made `url` column nullable

#### Schema (`schema.prisma`)
- Updated `LiteLLM_MCPServerTable` model to include stdio fields
- Made `url` optional

### 3. Frontend Changes

#### Create MCP Server (`ui/litellm-dashboard/src/components/mcp_tools/create_mcp_server.tsx`)
- Added "Standard I/O (stdio)" option to transport selector
- Added conditional rendering for stdio-specific fields:
  - Command input field
  - Dynamic arguments list (with add/remove functionality)
  - Dynamic environment variables list (key-value pairs)
- Made URL field conditional (only shown for HTTP/SSE transports)
- Updated form validation to require stdio fields when stdio transport is selected
- Enhanced payload transformation to handle stdio configuration

#### Edit MCP Server (`ui/litellm-dashboard/src/components/mcp_tools/mcp_server_edit.tsx`)
- Added stdio option to transport selector

#### Types (`ui/litellm-dashboard/src/components/mcp_tools/types.tsx`)
- Added `STDIO: "stdio"` to `TRANSPORT` constants

## Usage Examples

### Backend Usage

```python
from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPTransport

# Create stdio MCP client
stdio_config = {
    "command": "npx",
    "args": ["-y", "@circleci/mcp-server-circleci"],
    "env": {
        "CIRCLECI_TOKEN": "your-token",
        "CIRCLECI_BASE_URL": "https://circleci.com"
    }
}

client = MCPClient(
    transport_type=MCPTransport.stdio,
    stdio_config=stdio_config
)

# Connect and use the client
async with client:
    tools = await client.list_tools()
    # Use tools...
```

### Frontend Usage

1. Select "Standard I/O (stdio)" from Transport Type dropdown
2. Enter command (e.g., "npx")
3. Add arguments (e.g., "-y", "@circleci/mcp-server-circleci")
4. Add environment variables as key-value pairs
5. Submit form

### API Request Example

```json
{
  "alias": "circleci_mcp_server",
  "description": "CircleCI MCP Server",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@circleci/mcp-server-circleci"],
  "env": {
    "CIRCLECI_TOKEN": "your-circleci-token",
    "CIRCLECI_BASE_URL": "https://circleci.com"
  },
  "spec_version": "2025-03-26",
  "auth_type": "none",
  "mcp_access_groups": ["admin"]
}
```

## Validation Rules

### For stdio transport:
- `command` is required
- `args` is required (can be empty array)
- `env` is optional
- `url` is ignored/not required

### For HTTP/SSE transport:
- `url` is required
- `command`, `args`, `env` are ignored

## Benefits

1. **Broader MCP Server Support**: Can now connect to any MCP server that runs as a command-line process
2. **Local Development**: Easier to develop and test MCP servers locally
3. **Package Managers**: Direct support for npm packages, pip packages, etc.
4. **Environment Control**: Full control over environment variables for the MCP server process
5. **Security**: Processes run in isolated environments

## Compatibility

- Fully backward compatible with existing HTTP and SSE MCP servers
- No breaking changes to existing functionality
- All existing MCP server configurations continue to work unchanged

## Implementation Status

✅ Backend MCP client stdio support  
✅ Type definitions and validation  
✅ Database schema updates  
✅ Frontend UI for stdio configuration  
✅ Request/response handling  
✅ Documentation and examples  

The implementation is complete and ready for use. Users can now configure stdio MCP servers through both the API and the web interface.