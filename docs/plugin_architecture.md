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
| `POST /api/plugin-auth` | public | Decrypts the encrypted litellm token for seamless sign-in |

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

Receives `{ "encrypted_token": "<fernet-ciphertext>" }`.

Decrypt with the same `LITELLM_SALT_KEY` your litellm proxy uses:

```python
import base64, hashlib, os
from cryptography.fernet import Fernet

def make_cipher():
    key = base64.urlsafe_b64encode(hashlib.sha256(os.environ["LITELLM_SALT_KEY"].encode()).digest())
    return Fernet(key)

def plugin_auth(encrypted_token: str) -> str:
    return make_cipher().decrypt(encrypted_token.encode()).decode()
```

Return `{ "token": "<decrypted-litellm-token>" }`.  The plugin's browser
client stores this token and uses it to authenticate API calls back to litellm
via the `/plugin-proxy/my-plugin/*` reverse proxy.

---

## How iframe auth works

```
litellm UI
  ├─ GET /api/plugins/auth-token        → { encrypted_token }
  └─ postMessage({ type:"litellm-auth", encrypted_token }, pluginOrigin)
       │
       ▼
Plugin iframe browser
  └─ POST /api/plugin-auth { encrypted_token }
       │
       ▼
Plugin server
  ├─ decrypt(encrypted_token, LITELLM_SALT_KEY) → raw litellm token
  └─ return { token }  →  stored in sessionStorage
```

The raw litellm token never appears in plaintext outside the proxy process.
A postMessage intercept yields only useless ciphertext.

---

## Proxy routes (admin only)

- `GET /api/plugins` — list registered plugins (returns `plugin_key` to `proxy_admin` only)
- `GET /api/plugins/auth-token` — encrypted caller token for iframe auth
- `ANY /plugin-proxy/{name}/{path}` — authenticated reverse proxy; strips caller credentials, injects `plugin_key`

---

## Security checklist

- [ ] `LITELLM_SALT_KEY` is set and shared with the plugin service
- [ ] `plugin_key` is a dedicated credential scoped to the plugin (not your litellm master key)
- [ ] Plugin's `POST /api/plugin-auth` validates the decrypted token (e.g. checks it with litellm `/key/info`)
- [ ] Plugin service URL uses HTTPS in production
