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

A trusted web application can request access through the gateway's configured SSO provider. Proxy-API OAuth grants accept loopback redirects by default. Hosted access requires shared Redis and an exact callback URI configured on every gateway replica:

```shell
LITELLM_PROXY_API_OAUTH_REDIRECT_URIS=https://reports.example.com/oauth/callback
LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS=https://admin.example.com/oauth/callback
```

Multiple URIs are comma-separated. Each hosted URI must use HTTPS and contain no userinfo, query string, or fragment. Matching includes the complete path and port; wildcards and subdomain matching are not supported. Only register callbacks operated by trusted applications

The application registers the same URI at `/register` and uses `/authorize` with `resource` set to the gateway base URL, S256 PKCE, and state. LiteLLM handles SSO and presents an approval page with team selection before returning an authorization code. The application exchanges the code at `/token` using the original verifier and callback URI. Applications must validate state and bind the callback to the browser that started sign-in

The default `proxy:read` scope permits only GET requests to `/models`, `/v1/models`, `/global/activity`, `/global/activity/model`, `/global/spend`, `/global/spend/provider`, and `/global/spend/report`. Global reports also require the user's current proxy administrator or administrator viewer role. The selected team records attribution and requires current membership; it does not narrow an administrator's global reporting permissions. Credential/configuration endpoints, request contents, writes, and LLM calls are denied

Applications needing administrative access must explicitly request `scope=proxy:admin`. Only callbacks in `LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS` can receive it, and the consenting user must remain a full `proxy_admin`. This scope permits gateway management, usage, and model-inference APIs through their existing authorization checks. Authentication, credential delegation, vault, passthrough, and master-key-only routes are excluded. The consent screen lists these powers and model charges. Scope is fixed at consent and cannot increase on refresh

Discover support using `/.well-known/litellm-cli-auth`. Its `hosted_app` object advertises enabled scopes and token lifetimes; clients should check that contract instead of a gateway version or image tag. Configure the gateway URL and trusted callback once, then use normal gateway releases

Hosted access tokens expire after five minutes. Each user/application grant expires 24 hours after consent, including all refreshes. Refresh rotates both tokens; clients must serialize refresh requests and replace their saved pair. Already-issued access tokens remain valid until their original expiry so admitted work can finish. Do not retry an uncertain refresh; reconnect instead. A grant permits at most 512 refreshes, bounding replay-detection storage. Replaying a spent refresh token revokes the entire grant, including current API access. Send the current or spent refresh token, or an issued access token, and its `client_id` to `POST /revoke` when disconnecting. Demoting an administrative grant holder, deleting a user, deactivating them, removing their selected team membership, or blocking that team also stops access

Every authorized request reads the grant from shared Redis and the current user/team from the database. There is no local-cache or stale-permission fallback. Missing grants, Redis loss, database errors, and removed callback trust deny access. Callback configuration changes must reach every replica before removal is effective across the deployment. Grant records are separated by the gateway master key; rotating it invalidates hosted grants. Configure `PROXY_BASE_URL` consistently across replicas and use the same Redis namespace

Hosted issuance, refresh, and API access are disabled whenever `custom_auth` is configured or `enable_oauth2_auth` or `enable_oauth2_proxy_auth` is enabled. A custom hook also disables hosted access when its enterprise mode is `off`. Arbitrary custom or external identity and permission rules cannot safely be combined with these grants. Revocation remains available, including with expired or rotated access tokens, so existing grants can still be disconnected. Enabling an exclusive auth mode denies existing hosted grants; disabling it can restore unexpired grants that were not revoked

The hosted application's server must retain the PKCE verifier, verify OAuth state on return, exchange the code server-side, and keep both tokens out of browser storage. Give the browser a Secure, HttpOnly session cookie with CSRF protection. Its logout/disconnect handler must call `/revoke` before discarding the server session. The gateway never sends an access token in a redirect URL

Native loopback CLI grants and MCP grants retain their existing permissions. Hosted clients cannot use the native IdP token-exchange grant. Refresh tokens from earlier builds that issued personal credentials to hosted callbacks require a new sign-in. Previously issued personal access credentials cannot be distinguished from CLI credentials and remain valid until their original expiry; rotate the gateway's credential encryption key if they must be invalidated immediately. Removing the image override alone does not revoke those old credentials


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes
