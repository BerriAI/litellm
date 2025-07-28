# Custom HTTP Client

LiteLLM supports custom `httpx.Client` objects for HTTP control across providers.

## Quick Start

### Per-Request Client

```python
import httpx
import litellm

# Custom client with proxy/SSL settings
client = httpx.Client(
    timeout=30.0,
    verify=False,
    proxy="http://proxy.example.com:8080"
)

response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    client=client
)
```

### Global Client Session

```python
import httpx
import litellm

# Set global client for all requests
litellm.client_session = httpx.Client(timeout=60.0)

# All calls use the global client
response = litellm.completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Async Client

```python
import httpx
import litellm

async def main():
    async_client = httpx.AsyncClient(timeout=30.0)
    
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        client=async_client
    )
    
    await async_client.aclose()  # Important: close async clients

# Global async client
litellm.aclient_session = httpx.AsyncClient(timeout=60.0)
```

## Proxy Server Usage

When using LiteLLM proxy server, set the custom client before starting:

```python
import httpx
import litellm

# Configure custom client for proxy
litellm.client_session = httpx.Client(
    proxy="http://corporate-proxy.com:8080",
    verify="/path/to/ca-bundle.crt"
)

# Start proxy server - will use custom client for all upstream requests
litellm.run_server(host="0.0.0.0", port=4000)
```

## Common Use Cases

### Corporate Proxy
```python
client = httpx.Client(proxies={"http://": "http://proxy.com:8080"})
response = litellm.completion(model="gpt-3.5-turbo", messages=[...], client=client)
```

### Custom SSL/TLS
```python
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
client = httpx.Client(verify=ssl_context)
```

### Request Logging
```python
class LoggingClient(httpx.Client):
    def send(self, request, **kwargs):
        print(f"{request.method} {request.url}")
        return super().send(request, **kwargs)
```

## Provider Support

- **OpenAI** ✅ Full support via SDK wrapping
- **Anthropic** ⚠️ Limited support (async only) 
- **Azure OpenAI** ❌ Not yet supported
- **Gemini** ✅ Direct support
- **Others** ❌ Coming soon

## Limitations

- Some providers require specific SDK clients and don't support raw httpx
- Streaming may not work with all custom clients
- Connection pooling settings may not transfer to provider SDKs

## Notes

- Global sessions (`client_session`, `aclient_session`) are fallbacks
- Always close async clients: `await client.aclose()`
- Provider SDKs take precedence when passed directly 