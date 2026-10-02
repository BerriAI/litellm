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


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes

## Hosted admin applications

A trusted application can reuse the gateway's configured SSO, consent and PKCE flow. Configure `PROXY_BASE_URL` and `LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS=https://admin.example/oauth/callback` on every gateway and backend instance, with shared Redis and signing configuration. Callbacks match exactly, require HTTPS, and cannot contain userinfo, queries or fragments

Clients discover `hosted_app` at `/.well-known/litellm-cli-auth`, register their callback, and request `scope=proxy:admin` with the advertised resource. Only current full proxy admins can consent. The existing `/token` endpoint returns signed sessions bound to the user, selected team, client, callback and gateway. Access lasts five minutes; renewal ends 24 hours after consent

Every request checks the shared session and current database permissions, then uses the gateway's normal authorization and budget checks. Admin API operations and model calls are permitted; MCP admission, login flows, personal credential stores, key, user and invitation issuance and master-key-only routes are excluded. Custom auth or external OAuth API authentication disables app issuance, renewal and access

The existing `/revoke` endpoint disconnects all access and refresh generations, including when called with an expired access token before the session deadline. Refresh replay revokes the session. Missing Redis or database authority denies access; failed revocation is retryable. Keep tokens on the application server and validate OAuth state before exchanging the code

Upgrade all serving components before enabling the callback. Older releases reject hosted callbacks or the added signed claim. Clients use discovery and a stable gateway URL; they do not require a separately pinned backend image

Native IdP token exchange and refresh remain supported for loopback-only, HTTPS-only and mixed callback registrations. Selecting a loopback callback for authorization-code consent retains native behavior. Credentials issued by older patched releases in native or CLI format are indistinguishable from native credentials and retain existing lifetime, renewal and revocation policy. New hosted app sessions require explicit admin consent and use the fixed session deadline
