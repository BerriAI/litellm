# LiteLLM Plugin Architecture

Plugins let external services appear as selectable modes in the litellm UI sidebar alongside the AI Gateway.

---

## Quick start

### 1. Configure the plugin

Add a `plugins` block to your litellm `config.yaml`:

```yaml
general_settings:
  master_key: sk-...
  plugins:
    - name: my-plugin            # unique identifier (no spaces)
      display_name: My Plugin    # shown in the UI dropdown
      url: "https://my-plugin.example.com"
      plugin_key: "sk-..."       # plugin's own auth credential
```

`plugin_key` is injected as `Authorization: Bearer <plugin_key>` on every
request proxied through `/plugin-proxy/my-plugin/*`.  The caller's litellm
credential is stripped before forwarding so the plugin never receives a live
litellm API key.

### 2. Implement two endpoints on your service

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /api/plugin-manifest` | public | Returns plugin metadata for the UI |
| `POST /api/plugin-auth` | public | Decrypts the identity claim for seamless sign-in |

#### `GET /api/plugin-manifest`

```json
{
  "name": "my-plugin",
  "display_name": "My Plugin",
  "version": "1.0.0",
  "nav_items": [
    { "key": "home",    "label": "Home",    "icon": "HomeOutlined",    "path": "/" },
    { "key": "reports", "label": "Reports", "icon": "BarChartOutlined", "path": "/reports" }
  ],
  "capabilities": ["reports", "data"]
}
```

#### `POST /api/plugin-auth`

Receives `{ "session_claim": "<fernet-ciphertext>" }`.

The proxy never shares `LITELLM_SALT_KEY` with your plugin.  Each plugin is
provisioned with its own dedicated key, derived as
`HMAC-SHA256(LITELLM_SALT_KEY, plugin_name)`.  Compute it once on the proxy
host and hand the result to your plugin as a secret (e.g. `PLUGIN_AUTH_KEY`):

```bash
python -c 'import base64,hmac,hashlib,os; \
print(base64.urlsafe_b64encode(hmac.new(os.environ["LITELLM_SALT_KEY"].encode(), b"my-plugin", hashlib.sha256).digest()).decode())'
```

A compromised plugin holding only this scoped key cannot recover
`LITELLM_SALT_KEY` or decrypt any other litellm secret.

Decrypt and validate the claim with that key:

```python
import json, os, time
from cryptography.fernet import Fernet

_CLAIM_TTL_SECONDS = 30

def plugin_auth(session_claim: str) -> dict:
    cipher = Fernet(os.environ["PLUGIN_AUTH_KEY"].encode())
    claim = json.loads(cipher.decrypt(session_claim.encode(), ttl=_CLAIM_TTL_SECONDS))
    if claim.get("plugin") != "my-plugin":
        raise ValueError("claim audience mismatch")
    if int(claim.get("exp", 0)) < int(time.time()):
        raise ValueError("claim expired")
    return claim
```

The claim is `{ "plugin", "user_id", "user_role", "exp" }`; it carries no
litellm bearer token.  Establish the plugin's own session from `user_id` /
`user_role` and authenticate API calls back to litellm through the
`/plugin-proxy/my-plugin/*` reverse proxy, which injects `plugin_key` for you.

---

## How iframe auth works

```
litellm UI
  ├─ GET /api/plugins/auth-token        -> { session_claim }
  └─ postMessage({ type:"litellm-auth", session_claim }, pluginOrigin)
       │
       ▼
Plugin iframe browser
  └─ POST /api/plugin-auth { session_claim }
       │
       ▼
Plugin server
  ├─ decrypt(session_claim, PLUGIN_AUTH_KEY) -> { user_id, user_role, exp }
  └─ establish plugin session  ->  stored in sessionStorage
```

No litellm bearer token ever leaves the proxy; the claim only conveys the
caller's identity and expires after 30 seconds.  A postMessage intercept
yields ciphertext that is useless without the plugin's scoped key.

---

## Proxy routes

- `GET /api/plugins` — list registered plugins (`name`, `display_name`, `url`). `plugin_key` is **never** returned; it stays server-side. Requires an authenticated caller.
- `GET /api/plugins/auth-token?plugin_name=<name>` — short-lived encrypted identity claim for the named plugin. Requires `LITELLM_SALT_KEY` to be set (503 otherwise) and the plugin to be registered (404 otherwise).
- `ANY /plugin-proxy/{name}/{path}` — authenticated reverse proxy to the plugin backend. Restricted to `proxy_admin`.

---

## Reverse proxy behaviour

When an admin (or server-to-server caller) hits `/plugin-proxy/<name>/<path>`, the proxy authenticates the caller locally, then rewrites the request before forwarding it to the plugin's `url`:

- **Every litellm credential header is stripped** — `Authorization`, `x-api-key`, `API-Key`, `x-goog-api-key`, `Ocp-Apim-Subscription-Key`, `x-litellm-api-key`, any configured `litellm_key_header_name`, plus `Cookie`. The plugin can never be handed the caller's live litellm key.
- **`plugin_key` is injected** as `Authorization: Bearer <plugin_key>` — the only credential the plugin receives.
- **Caller identity is forwarded** as `x-litellm-user-id` and `x-litellm-user-role` so the plugin can run its own authorization. These are informational, not credentials.
- **Responses are sandboxed** — `Content-Security-Policy: sandbox` and `X-Content-Type-Options: nosniff` are set so plugin-controlled bytes served from the litellm origin cannot execute against the dashboard.

---

## Security checklist

- [ ] `LITELLM_SALT_KEY` is set on the proxy and never shared with the plugin
- [ ] The plugin holds only its derived `HMAC(LITELLM_SALT_KEY, plugin_name)` key, provisioned as a dedicated secret
- [ ] `plugin_key` is a dedicated credential scoped to the plugin (not your litellm master key)
- [ ] Plugin's `POST /api/plugin-auth` enforces the claim's `plugin` audience and `exp` (30s TTL)
- [ ] Plugin treats `x-litellm-user-id` / `x-litellm-user-role` as identity hints, not as proof of authentication
- [ ] Plugin service URL uses HTTPS in production
