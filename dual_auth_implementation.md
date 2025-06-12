# Dual Authentication Implementation for MCP Servers

This document describes the implementation of dual authentication for MCP (Model Context Protocol) servers in LiteLLM, allowing for both system-level authentication and user-level authorization.

## Overview

The implementation supports two levels of authentication:

1. **System Authentication**: Uses `x-litellm-key` header for LiteLLM proxy authentication
2. **User Authorization**: Uses `Authorization` header for user-specific permissions that get passed through to MCP servers

## Architecture

```
LibreChat/Client -> LiteLLM MCP Gateway -> Atlassian MCP
                    x-litellm-key         Authorization: Bearer xxx
                    Authorization         (passed through)
```

## Implementation Details

### 1. REST API Changes

The `/mcp/tools/call` endpoint has been modified to:
- Continue using `user_api_key_auth` dependency for LiteLLM system authentication
- Extract the `Authorization` header from incoming requests
- Pass the authorization header through to MCP server manager

### 2. MCP Server Manager Updates

The `MCPServerManager.call_tool()` method now:
- Accepts an optional `user_authorization_header` parameter
- Logs when authorization headers are provided
- Documents current limitations with MCP SDK header support

### 3. Current Limitations

Due to limitations in the current MCP Python SDK:
- Authorization headers cannot be fully passed through to external MCP servers
- The implementation logs authorization headers but cannot send them to HTTP/SSE MCP servers
- This is a known limitation that will be resolved when the MCP SDK supports custom headers

## Usage Example

```bash
# Client request with dual authentication
curl -X POST "http://localhost:4000/mcp/tools/call" \
  -H "x-litellm-key: sk-1234" \
  -H "Authorization: Bearer user-token-5678" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "create_jira_ticket",
    "arguments": {
      "summary": "New ticket",
      "project": "TEST"
    }
  }'
```

## Future Enhancements

When the MCP Python SDK supports custom headers:
1. Update `custom_streamablehttp_client_with_auth()` to properly pass headers
2. Implement SSE client header support
3. Remove warning messages about header limitations
4. Update documentation

## Files Modified

1. `/workspace/litellm/proxy/_experimental/mcp_server/server.py`
   - Modified `call_tool_rest_api()` to extract Authorization header
   - Updated `call_mcp_tool()` to accept user authorization header
   - Updated `_handle_managed_mcp_tool()` signature

2. `/workspace/litellm/proxy/_experimental/mcp_server/mcp_server_manager.py`
   - Modified `call_tool()` to accept and log authorization headers
   - Added warnings about current MCP SDK limitations

## Testing

To test the dual authentication:

1. Set up LiteLLM proxy with MCP servers configured
2. Send requests with both `x-litellm-key` and `Authorization` headers
3. Check logs for authorization header detection and warnings
4. Verify that LiteLLM authentication still works properly

## Security Considerations

- The `x-litellm-key` is validated by LiteLLM's authentication system
- The `Authorization` header is extracted and logged but not currently validated by LiteLLM
- Authorization validation should be handled by the target MCP servers
- Ensure proper logging levels to avoid exposing sensitive headers in production