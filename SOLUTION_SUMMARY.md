# SSL Certificate Verification Error - Solution Summary

## Issue
You were experiencing SSL certificate verification errors when registering MCP servers:
```
httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Missing Authority Key Identifier (_ssl.c:1032)
```

Even though:
- The proxy successfully connected to Vertex AI in GCP
- You tried setting multiple environment variables (HTTPX_CA_BUNDLE, REQUESTS_CA_BUNDLE, SSL_CERT_FILE, etc.)
- You attempted patching Python's certifi package

## Root Cause
The LiteLLM proxy's MCP server configuration didn't support passing custom SSL verification settings through to the underlying httpx client. The `ssl_verify` parameter existed in the MCPClient but wasn't exposed through the server configuration.

## Solution Implemented

I've implemented a complete fix that adds `ssl_verify` support throughout the MCP server configuration stack.

### Changes Made

1. **Added `ssl_verify` field to MCPServer model**
   - File: `litellm/types/mcp_server/mcp_server_manager.py`
   - Supports: `False` (disable SSL), string path to CA bundle, or `None` (use defaults)

2. **Updated MCP server manager to pass ssl_verify through**
   - File: `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py`
   - Modified: `_create_mcp_client()`, `load_servers_from_config()`, `build_mcp_server_from_table()`

3. **Added database support**
   - File: `litellm/proxy/schema.prisma` - Added `ssl_verify` column
   - File: `litellm/proxy/_types.py` - Added field to `LiteLLM_MCPServerTable`

4. **Added comprehensive tests**
   - File: `tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py`
   - Tests verify ssl_verify flows from config to client

## How to Use

### Option 1: YAML Configuration (Recommended)

Update your `config.yaml` to include `ssl_verify` for each MCP server:

```yaml
mcp_servers:
  my_mcp_server:
    url: "https://mcp.example.com"
    transport: "sse"
    auth_type: "bearer_token"
    authentication_token: "your-token"
    ssl_verify: "/etc/ssl/certs/ca-certificates.crt"  # Use your CA bundle path
```

Or disable SSL verification for development (NOT recommended for production):

```yaml
mcp_servers:
  dev_mcp_server:
    url: "https://dev-mcp.example.com"
    transport: "sse"
    ssl_verify: false  # ⚠️ Development only!
```

### Option 2: Environment Variable (Global Fallback)

If you don't set `ssl_verify` in your server config, it will fall back to the `SSL_CERT_FILE` environment variable:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

This is the environment variable that LiteLLM's SSL configuration respects (not HTTPX_CA_BUNDLE or REQUESTS_CA_BUNDLE).

### Option 3: Docker/Kubernetes

**Docker Compose:**
```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:latest
    volumes:
      - ./config.yaml:/app/config.yaml
      - /etc/ssl/certs:/etc/ssl/certs:ro  # Mount certificates
    environment:
      - SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

**Kubernetes:**
```yaml
# ConfigMap with CA bundle
apiVersion: v1
kind: ConfigMap
metadata:
  name: ca-bundle
data:
  ca-certificates.crt: |
    -----BEGIN CERTIFICATE-----
    ... your certificates ...
    -----END CERTIFICATE-----

---
# Deployment
spec:
  template:
    spec:
      containers:
      - name: litellm
        volumeMounts:
        - name: ca-bundle
          mountPath: /app/certs
        env:
        - name: SSL_CERT_FILE
          value: /app/certs/ca-certificates.crt
      volumes:
      - name: ca-bundle
        configMap:
          name: ca-bundle
```

## Migration Path

### Before (Workarounds - no longer needed):
```bash
# These no longer work or are needed:
export HTTPX_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt  # Not supported
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt  # Not used by httpx
python -c "import certifi; ..."  # No longer needed
```

### After (Proper Solution):

**Option A - Per-server in config.yaml:**
```yaml
mcp_servers:
  my_server:
    url: "https://mcp.example.com"
    ssl_verify: "/etc/ssl/certs/ca-certificates.crt"
```

**Option B - Global via environment:**
```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## Testing Your Configuration

1. **Verify the CA bundle path exists:**
   ```bash
   ls -l /etc/ssl/certs/ca-certificates.crt
   ```

2. **Test with a simple MCP server config:**
   ```yaml
   mcp_servers:
     test_server:
       url: "https://your-mcp-server.com"
       transport: "sse"
       ssl_verify: "/etc/ssl/certs/ca-certificates.crt"
   ```

3. **Check the logs** for SSL-related messages when starting the proxy

4. **Try registering the MCP server** - the SSL error should now be resolved

## Troubleshooting

### Still getting "Missing Authority Key Identifier"?

This specific error means the certificate chain is incomplete. Solutions:

1. **Ensure your CA bundle contains the full chain:**
   - Root CA certificate
   - Intermediate CA certificates
   - Your server certificate

2. **Combine certificates if needed:**
   ```bash
   cat server.crt intermediate.crt rootCA.crt > ca-bundle.crt
   ```

3. **Update system CA certificates:**
   ```bash
   # Debian/Ubuntu
   sudo update-ca-certificates
   
   # RHEL/CentOS
   sudo update-ca-trust
   ```

### Different certificate locations by OS:

- **Debian/Ubuntu:** `/etc/ssl/certs/ca-certificates.crt`
- **RHEL/CentOS/Fedora:** `/etc/pki/tls/certs/ca-bundle.crt`
- **Alpine Linux:** `/etc/ssl/cert.pem`
- **macOS:** Use the system keychain or provide custom bundle

## Database Migration

If you're using a database-backed proxy, you'll need to run a migration:

```bash
# Navigate to proxy directory
cd litellm/proxy

# Generate and run migration
prisma migrate dev --name add_ssl_verify_to_mcp_servers

# Or for production
prisma migrate deploy
```

Then update existing MCP servers via the API or directly in the database to add the `ssl_verify` field.

## Next Steps

1. **Update your configuration** with the `ssl_verify` parameter
2. **Test the connection** to your MCP server
3. **Remove any workarounds** (certifi patching, unsupported env vars)
4. **Document your configuration** for your team

## Example Configurations

A complete example configuration file is available at: `mcp_ssl_config_example.yaml`

## Files Modified

- `litellm/types/mcp_server/mcp_server_manager.py` - MCPServer model
- `litellm/proxy/_types.py` - Database model
- `litellm/proxy/schema.prisma` - Database schema
- `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py` - Server manager
- `tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py` - Tests

## Additional Documentation

For detailed information, see: `MCP_SSL_CERTIFICATE_FIX.md`

---

**This solution provides a proper, supported way to configure SSL verification for MCP servers without requiring workarounds or patches to Python packages.**
