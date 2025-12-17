# MCP SSL Certificate Verification Fix

## Problem

When registering MCP servers with the LiteLLM proxy, users were encountering SSL certificate verification errors:

```
httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Missing Authority Key Identifier (_ssl.c:1032)
```

This occurred even when:
- The proxy successfully connected to other services (like Vertex AI in GCP)
- Various SSL certificate bundle environment variables were set
- Custom CA bundles were provided

## Root Cause

The `MCPServer` model and `MCPServerManager._create_mcp_client()` method did not support passing custom SSL verification settings to the underlying `MCPClient`. This meant:

1. MCP servers could not be configured with custom CA bundles
2. SSL verification could not be disabled for development/testing
3. The `ssl_verify` parameter was not passed through the server configuration chain

## Solution

Added `ssl_verify` support throughout the MCP server configuration chain:

### 1. Updated Data Models

**File: `litellm/types/mcp_server/mcp_server_manager.py`**
- Added `ssl_verify: Optional[Union[bool, str]]` field to `MCPServer` model
- Can be `False` to disable SSL verification
- Can be a string path to a custom CA bundle file
- Can be `None` to use default SSL configuration

**File: `litellm/proxy/_types.py`**
- Added `ssl_verify: Optional[Union[bool, str]]` field to `LiteLLM_MCPServerTable` model
- Ensures database model matches the MCPServer model

**File: `litellm/proxy/schema.prisma`**
- Added `ssl_verify String?` field to database schema
- Allows storing SSL configuration in the database

### 2. Updated Server Manager

**File: `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py`**

- Modified `_create_mcp_client()` to pass `ssl_verify` to MCPClient:
  ```python
  return MCPClient(
      server_url=server_url,
      transport_type=transport,
      auth_type=server.auth_type,
      auth_value=mcp_auth_header or server.authentication_token,
      timeout=60.0,
      extra_headers=extra_headers,
      ssl_verify=server.ssl_verify,  # NEW: Pass ssl_verify
  )
  ```

- Updated `load_servers_from_config()` to read `ssl_verify` from config:
  ```python
  new_server = MCPServer(
      ...
      ssl_verify=server_config.get("ssl_verify", None),
  )
  ```

- Updated `build_mcp_server_from_table()` to read `ssl_verify` from database:
  ```python
  new_server = MCPServer(
      ...
      ssl_verify=getattr(mcp_server, "ssl_verify", None),
  )
  ```

### 3. Added Tests

**File: `tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py`**

Added two new test cases:
1. `test_ssl_verify_passed_to_mcp_client()` - Verifies ssl_verify is passed from MCPServer to MCPClient
2. `test_load_servers_from_config_with_ssl_verify()` - Verifies ssl_verify is loaded from YAML config

## Usage

### Option 1: YAML Configuration

Add `ssl_verify` to your MCP server configuration:

```yaml
mcp_servers:
  my_mcp_server:
    url: "https://mcp.example.com"
    transport: "sse"
    ssl_verify: false  # Disable SSL verification (for development only!)
    
  my_secure_mcp_server:
    url: "https://secure-mcp.example.com"
    transport: "http"
    ssl_verify: "/etc/ssl/certs/ca-certificates.crt"  # Use custom CA bundle
    
  my_default_mcp_server:
    url: "https://default-mcp.example.com"
    transport: "sse"
    # ssl_verify not specified - uses system defaults
```

### Option 2: Environment Variables (Fallback)

If `ssl_verify` is not set in the server config, the MCPClient will fall back to:
1. `SSL_VERIFY` environment variable
2. `SSL_CERT_FILE` environment variable
3. System default CA bundle (via certifi)

Example:
```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### Option 3: Database/API Configuration

When creating MCP servers via API or database, include the `ssl_verify` field:

```python
mcp_server = LiteLLM_MCPServerTable(
    server_id="my-server",
    server_name="my_mcp_server",
    url="https://mcp.example.com",
    transport="sse",
    ssl_verify="/path/to/ca-bundle.crt",  # or False, or None
)
```

## How SSL Verification Works

The SSL configuration flows through these layers:

```
MCPServer.ssl_verify
    ↓
MCPServerManager._create_mcp_client(server.ssl_verify)
    ↓
MCPClient.__init__(ssl_verify=...)
    ↓
MCPClient._create_httpx_client_factory()
    ↓
get_ssl_configuration(ssl_verify)
    ↓
httpx.AsyncClient(verify=ssl_config)
```

The `get_ssl_configuration()` function (in `litellm/llms/custom_httpx/http_handler.py`) handles:
1. Converting string paths to SSL contexts
2. Checking environment variables
3. Using certifi CA bundle as fallback
4. Creating cached SSL contexts for performance

## Security Considerations

⚠️ **WARNING**: Disabling SSL verification (`ssl_verify: false`) should only be used in development/testing environments. In production, always use proper SSL certificates and verification.

For self-signed certificates or internal CAs:
1. Add your CA certificate to a bundle file
2. Set `ssl_verify: "/path/to/ca-bundle.crt"`
3. Ensure the bundle includes the full certificate chain

## Troubleshooting

### "Missing Authority Key Identifier" Error

This error indicates the certificate chain is incomplete. Solutions:

1. **Use a complete CA bundle:**
   ```yaml
   ssl_verify: "/etc/ssl/certs/ca-certificates.crt"
   ```

2. **Update system CA certificates:**
   ```bash
   # Debian/Ubuntu
   sudo update-ca-certificates
   
   # RHEL/CentOS
   sudo update-ca-trust
   ```

3. **Include intermediate certificates:**
   Create a bundle with your certificate + intermediate certificates + root CA

4. **For development only - disable verification:**
   ```yaml
   ssl_verify: false  # NOT RECOMMENDED FOR PRODUCTION
   ```

### Environment Variables Not Working

The environment variable fallback only works if `ssl_verify` is not explicitly set in the server config. To use environment variables:

1. Don't set `ssl_verify` in your YAML config
2. Set `SSL_CERT_FILE` environment variable:
   ```bash
   export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
   ```

### Docker/Kubernetes Environments

When running in containers:

1. **Mount CA certificates:**
   ```yaml
   volumes:
     - /etc/ssl/certs:/etc/ssl/certs:ro
   ```

2. **Configure in YAML:**
   ```yaml
   mcp_servers:
     my_server:
       ssl_verify: "/etc/ssl/certs/ca-certificates.crt"
   ```

3. **Or set environment variable:**
   ```yaml
   env:
     - name: SSL_CERT_FILE
       value: /etc/ssl/certs/ca-certificates.crt
   ```

## Migration Guide

If you were previously working around this issue:

### Before
```bash
# Patching certifi (no longer needed)
python -c "import certifi; import shutil; shutil.copy('/etc/ssl/certs/ca-certificates.crt', certifi.where())"

# Setting httpx-specific env vars (not supported)
export HTTPX_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### After
```yaml
# Option 1: In config.yaml
mcp_servers:
  my_server:
    url: "https://mcp.example.com"
    ssl_verify: "/etc/ssl/certs/ca-certificates.crt"

# Option 2: Or use environment variable
# (don't set ssl_verify in config)
```
```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## Database Migration

If you have existing MCP servers in your database, they will continue to work with default SSL settings. To update them:

1. Run Prisma migration to add the `ssl_verify` column:
   ```bash
   prisma migrate dev --name add_ssl_verify_to_mcp_servers
   ```

2. Update existing servers via API or database:
   ```python
   await prisma.litellm_mcpservertable.update(
       where={"server_id": "my-server-id"},
       data={"ssl_verify": "/path/to/ca-bundle.crt"}
   )
   ```

## Testing

Run the test suite to verify SSL configuration:

```bash
pytest tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py::TestMCPServerManager::test_ssl_verify_passed_to_mcp_client -v
pytest tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py::TestMCPServerManager::test_load_servers_from_config_with_ssl_verify -v
pytest tests/test_litellm/experimental_mcp_client/test_mcp_client.py::TestMCPClient::test_mcp_client_ssl_verify_parameter -v
```

## Summary of Changes

Files modified:
1. `litellm/types/mcp_server/mcp_server_manager.py` - Added ssl_verify field
2. `litellm/proxy/_types.py` - Added ssl_verify field to database model
3. `litellm/proxy/schema.prisma` - Added ssl_verify column to database schema
4. `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py` - Pass ssl_verify through configuration chain
5. `tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py` - Added test coverage

This fix provides a proper, supported way to configure SSL verification for MCP servers without requiring workarounds or patches.
