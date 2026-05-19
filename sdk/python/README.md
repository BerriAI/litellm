# xct-litellm

Python SDK for the xct-litellm capability provider.

```python
from xct_litellm import XctClient, PKCESession

xct = XctClient(
    base_url="https://api.xct.test",
    access_token="sk-xct-...",
    app_id="xct-chat",
)

# Discovery
caps = xct.capabilities.list()

# Chat (OpenAI-shape, both sync + async)
reply = xct.chat.completions.create(
    model="deepseek-v3.2",
    messages=[{"role": "user", "content": "hi"}],
)

# Streaming
for chunk in xct.chat.completions.create(
    model="deepseek-v3.2",
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
):
    print(chunk)

# Async equivalent
async def go():
    reply = await xct.chat.completions.acreate(
        model="deepseek-v3.2",
        messages=[{"role": "user", "content": "hi"}],
    )
    return reply

# A2A agent
res = xct.agents.invoke("agent_rosalind", message={"role": "user", "parts": [{"text": "..."}]})

# Server-side OAuth (refresh / token exchange)
sess = PKCESession(
    client_id="xct_abc",
    redirect_uri="https://chat.xct.test/oauth/callback",
)
print(sess.authorize_url("https://api.xct.test", scope="read write"))
# ... user clicks, your callback gets `code`
tokens = sess.complete("https://api.xct.test", code="abc")
# tokens["access_token"], tokens["refresh_token"]
```

## Surface

| Resource | Methods |
|---|---|
| `xct.capabilities` | `list()` / `alist()` |
| `xct.agents` | `list(...)` / `get(id)` / `invoke(id, message=..., stream=False)` |
| `xct.mcp` | `list_tools()` |
| `xct.skills` | `list(...)` / `get(id)` |
| `xct.chat.completions` | `create(**openai_payload)` (`acreate(...)` async) |
| `PKCESession` | `.authorize_url()`, `.complete()`, `.acomplete()` |

Every method has an `a`-prefixed async variant (`alist`, `acreate`, ...).

## Errors

All non-2xx responses raise an `XctError` subclass with `.status` and `.body`:

- 401 / 403 → `AuthError`
- 404 → `CapabilityNotFoundError`
- 429 → `RateLimitError`
- everything else → `XctError`

## Underlying transport

`httpx` only. Initialize with `http_client=` / `async_http_client=` to inject
your own (e.g. for retries, telemetry, mTLS).
