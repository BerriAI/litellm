# Quick Start Guide - Fix SSL Certificate Error for MCP Servers

## TL;DR - What You Need to Do

Your SSL certificate error when registering MCP servers is now fixed. Here's what you need to do:

### 1. Update Your Configuration

Add the `ssl_verify` parameter to your MCP server configuration in `config.yaml`:

```yaml
mcp_servers:
  your_mcp_server:
    url: "https://your-mcp-server.com"
    transport: "sse"
    auth_type: "bearer_token"
    authentication_token: "your-token"
    ssl_verify: "/etc/ssl/certs/ca-certificates.crt"  # ← ADD THIS LINE
```

### 2. Or Use Environment Variable

Instead of adding to config, you can set this globally:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

Then restart your LiteLLM proxy.

### 3. Verify the Fix

Start your proxy and try registering the MCP server. The SSL error should be gone.

## Why This Works

Before this fix:
- The `ssl_verify` parameter existed in MCPClient but wasn't exposed through server configuration
- You had to use workarounds that didn't actually work (HTTPX_CA_BUNDLE, certifi patching, etc.)

After this fix:
- The `ssl_verify` parameter flows from your config → MCPServer → MCPServerManager → MCPClient → httpx
- You can now configure SSL verification the proper way

## Common Use Cases

### Case 1: Corporate/Internal CA
```yaml
ssl_verify: "/path/to/your/corporate-ca-bundle.crt"
```

### Case 2: Self-Signed Certificate (Development)
```yaml
ssl_verify: false  # ⚠️ Development only!
```

### Case 3: Standard System CA Bundle
```yaml
ssl_verify: "/etc/ssl/certs/ca-certificates.crt"  # Most Linux systems
```

### Case 4: Let Environment Variable Handle It
```yaml
# Don't set ssl_verify in config
# Set environment variable instead:
# export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## Docker/Kubernetes Users

### Docker Compose
```yaml
services:
  litellm:
    volumes:
      - /etc/ssl/certs:/etc/ssl/certs:ro
    environment:
      - SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### Kubernetes
```yaml
env:
  - name: SSL_CERT_FILE
    value: /etc/ssl/certs/ca-certificates.crt
```

## Testing

1. Check your CA bundle exists:
   ```bash
   ls -l /etc/ssl/certs/ca-certificates.crt
   ```

2. Add `ssl_verify` to your config (see above)

3. Restart LiteLLM proxy

4. Try registering your MCP server - should work now!

## Still Having Issues?

### Error: "Missing Authority Key Identifier"

This means your certificate chain is incomplete. You need:
- Root CA certificate
- Any intermediate certificates
- Your server certificate

**Solution:** Create a complete bundle:
```bash
cat server.crt intermediate.crt root.crt > complete-bundle.crt
```

Then use:
```yaml
ssl_verify: "/path/to/complete-bundle.crt"
```

### Error: "No such file or directory"

The CA bundle path doesn't exist. Common paths by OS:

- **Ubuntu/Debian:** `/etc/ssl/certs/ca-certificates.crt`
- **RHEL/CentOS:** `/etc/pki/tls/certs/ca-bundle.crt`
- **Alpine Linux:** `/etc/ssl/cert.pem`

Use the correct path for your system.

### For Self-Signed Certificates

If you're testing with a self-signed certificate:

**Option 1 (Secure):** Add your CA to the bundle
```bash
# Copy your CA certificate
sudo cp your-ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates

# Then use system bundle
ssl_verify: "/etc/ssl/certs/ca-certificates.crt"
```

**Option 2 (Development only):** Disable verification
```yaml
ssl_verify: false
```

## Need More Details?

See the comprehensive documentation:
- `SOLUTION_SUMMARY.md` - Full solution explanation
- `MCP_SSL_CERTIFICATE_FIX.md` - Technical details and troubleshooting
- `mcp_ssl_config_example.yaml` - Complete configuration examples

## Summary of Changes

This fix adds proper `ssl_verify` support throughout the MCP server stack:
- ✅ Configure per MCP server in YAML
- ✅ Use environment variable as fallback
- ✅ Store in database for API-created servers
- ✅ Passes through to httpx client correctly
- ✅ No more workarounds needed!

---

**Your SSL certificate verification error should now be resolved. Just add `ssl_verify` to your config and restart!**
