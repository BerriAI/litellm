# litellm-proxy

A local, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

## usage 

```shell 
$ uv tool install litellm
```
```shell
$ litellm --model ollama/codellama 

#INFO: Ollama running on http://0.0.0.0:8000
```

## replace openai base
```python 
import openai # openai v1.0.0+
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:8000") # set proxy to base_url
# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
``` 

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/simple_proxy)


## Hosted applications using gateway SSO

A trusted web application can request a personal proxy-API session through the gateway’s configured SSO provider. By default, proxy-API OAuth grants only accept loopback redirects. To permit a hosted application, configure its exact callback URI on the gateway:

```shell
LITELLM_PROXY_API_OAUTH_REDIRECT_URIS=https://admin.example.com/oauth/callback
```

Multiple URIs are comma-separated. Each hosted URI must use HTTPS and contain no userinfo, query string, or fragment. Matching includes the complete path and port; wildcards and subdomain matching are not supported. Only allow callbacks operated by applications trusted to hold users’ personal proxy credentials

The application registers the same URI at `/register` and uses `/authorize` with `resource` set to the gateway base URL, S256 PKCE, and state. LiteLLM handles SSO and presents an approval page with team selection before returning an authorization code. The application exchanges the code at `/token` using the original verifier and callback URI. Applications must validate state and bind the callback to the browser that started sign-in

This setting does not grant an admin role or change existing native-client and MCP grants. The resulting credential uses the signed-in user’s permissions. Hosted applications must protect access and refresh tokens and handle expiry and revocation


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes
