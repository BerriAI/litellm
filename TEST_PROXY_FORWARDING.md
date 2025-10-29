# Testing X-Forwarded-* Header Support

This guide helps you test the X-Forwarded-Host and X-Forwarded-Proto fixes before deploying to production.

## Quick Test (Recommended)

### Step 1: Start LiteLLM Proxy

Start your LiteLLM proxy as you normally would with your MCP server configured:

```bash
litellm --config your_config.yaml --port 4000
```

Make sure you have at least one MCP server configured with OAuth (e.g., `github`).

### Step 2: Run the Test Script

```bash
python test_proxy_forwarding.py
```

This will:
1. Make requests to your LiteLLM instance with X-Forwarded-* headers
2. Verify that the OAuth discovery endpoints return the correct external URLs
3. Show you exactly what URLs are being returned

**Example output:**
```
======================================================================
Testing X-Forwarded-* Header Support
======================================================================

Internal URL: http://localhost:4000
External URL: https://proxy.example.com
MCP Server: github

Forwarded Headers: {
  "X-Forwarded-Proto": "https",
  "X-Forwarded-Host": "proxy.example.com"
}

======================================================================

[Test 1] OAuth Authorization Server Discovery
----------------------------------------------------------------------
Request: GET http://localhost:4000/.well-known/oauth-authorization-server/github
Status: 200

✓ Checking endpoints use external URL:
  ✓ authorization_endpoint: https://proxy.example.com/github/authorize
  ✓ token_endpoint: https://proxy.example.com/github/token
  ✓ registration_endpoint: https://proxy.example.com/github/register

[Test 2] OAuth Protected Resource Discovery
----------------------------------------------------------------------
Request: GET http://localhost:4000/.well-known/oauth-protected-resource/github/mcp
Status: 200

✓ Checking authorization servers use external URL:
  ✓ https://proxy.example.com/github

[Test 3] Client Registration Endpoint
----------------------------------------------------------------------
Request: POST http://localhost:4000/github/register
Status: 200

✓ Checking redirect_uris use external URL:
  ✓ https://proxy.example.com/callback
```

### Step 3: Customize for Your Setup

You can customize the test for your specific environment:

```bash
# Test with your actual external hostname
python test_proxy_forwarding.py \
  --external-host proxy.example.com \
  --mcp-server github

# Test with a different LiteLLM port
python test_proxy_forwarding.py \
  --litellm-url http://localhost:8000 \
  --external-host my-proxy.company.com \
  --mcp-server my_mcp_server
```

### Step 4: Verify with curl (Alternative)

You can also test manually with curl:

```bash
# Test OAuth Authorization Server Discovery
curl -H "X-Forwarded-Proto: https" \
     -H "X-Forwarded-Host: proxy.example.com" \
     http://localhost:4000/.well-known/oauth-authorization-server/github | jq

# Expected: All URLs should start with https://proxy.example.com

# Test OAuth Protected Resource Discovery
curl -H "X-Forwarded-Proto: https" \
     -H "X-Forwarded-Host: proxy.example.com" \
     http://localhost:4000/.well-known/oauth-protected-resource/github/mcp | jq

# Test Client Registration
curl -X POST \
     -H "X-Forwarded-Proto: https" \
     -H "X-Forwarded-Host: proxy.example.com" \
     -H "Content-Type: application/json" \
     http://localhost:4000/github/register | jq
```

## What to Look For

### ✅ Success Indicators

All URLs in the responses should use your **external** hostname and scheme:
- `https://proxy.example.com/...` ✓
- NOT `http://localhost:4000/...` ✗
- NOT `http://proxy.example.com:8888/...` ✗

### Common Issues

**Issue 1: URLs still show localhost**
- Cause: X-Forwarded headers not being sent
- Solution: Make sure your proxy sets the headers correctly

**Issue 2: URLs show http instead of https**
- Cause: X-Forwarded-Proto header not set
- Solution: Ensure `X-Forwarded-Proto: https` is set by your proxy

**Issue 3: URLs show internal port (e.g., :8888)**
- Cause: X-Forwarded-Host includes port but shouldn't
- Solution: Configure your proxy to only forward the hostname without port

## Example LiteLLM Config

Make sure your MCP server is configured in your LiteLLM config:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: ${OPENAI_API_KEY}

mcp_servers:
  - server_name: github
    auth_type: oauth2
    client_id: ${GITHUB_CLIENT_ID}
    client_secret: ${GITHUB_CLIENT_SECRET}
    authorization_url: https://github.com/login/oauth/authorize
    token_url: https://github.com/login/oauth/access_token
    scopes:
      - repo
      - user
```

## Testing End-to-End OAuth Flow

After verifying the URLs are correct with the test script, you can test the full OAuth flow:

1. Start LiteLLM proxy with your config
2. Configure Claude Desktop with your MCP server using the external URL:
   ```json
   {
     "mcpServers": {
       "github": {
         "url": "https://proxy.example.com/github/mcp"
       }
     }
   }
   ```
3. Try to connect - it should now work with the correct URLs!

## Troubleshooting

### Test script shows ✗ for URLs

If you see `✗` next to any URLs, the headers aren't being processed correctly. Check:
1. Is your LiteLLM running the latest code with the X-Forwarded-* fixes?
2. Are you passing the headers correctly in the test script?
3. Check the LiteLLM logs for any errors

### OAuth flow still fails in production

1. Run the test script pointing to your production LiteLLM instance
2. Verify your nginx/proxy is actually setting the X-Forwarded-* headers
3. Check nginx logs to see what headers are being sent
4. Use browser dev tools to inspect the OAuth redirect URLs

## Need Help?

- Check the LiteLLM logs: `tail -f litellm.log`
- Enable verbose logging: `litellm --config config.yaml --debug`
- Verify headers in nginx: Add to nginx config:
  ```nginx
  add_header X-Debug-Forwarded-Proto $http_x_forwarded_proto always;
  add_header X-Debug-Forwarded-Host $http_x_forwarded_host always;
  ```
