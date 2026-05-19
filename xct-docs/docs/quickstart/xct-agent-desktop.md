---
sidebar_position: 3
---

# Quickstart: xct-agent-desktop

xct-agent-desktop is the desktop client (Electron / Tauri) that lets a user
chat with **agents** directly. PKCE login + A2A streaming over SSE.

## What you'll build

- A native shell that opens the system browser for OAuth login
- Captures the callback in a localhost loopback redirect
- Lists agents the user has access to
- Streams agent invocations into a chat UI

## 1. Provision the app (admin)

In **XCT Apps** → **New app**:

| Field | Value |
|---|---|
| `app_name` | `xct-agent-desktop` |
| `redirect_uris` | `http://127.0.0.1:53682/oauth/callback` (loopback) |
| `default_scopes` | `read write agents:invoke` |

Loopback redirects are the standard pattern for native apps — RFC 8252.

## 2. Listen on a loopback port

```python
import http.server
import socketserver
import threading
import urllib.parse

PORT = 53682
captured: dict = {}

class CB(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        qs = urllib.parse.urlparse(self.path).query
        params = dict(urllib.parse.parse_qsl(qs))
        captured.update(params)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>You can close this window.</h2>")

httpd = socketserver.TCPServer(("127.0.0.1", PORT), CB)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
```

## 3. Open the system browser

```python
import webbrowser
from xct_litellm import PKCESession

session = PKCESession(
    client_id="xct_abc...",
    redirect_uri=f"http://127.0.0.1:{PORT}/oauth/callback",
)
webbrowser.open(session.authorize_url(
    "https://api.xct.test",
    scope="read write agents:invoke",
))

# wait for the loopback handler to fill `captured`
while "code" not in captured:
    time.sleep(0.1)

tokens = session.complete("https://api.xct.test", code=captured["code"])
# tokens["access_token"] + tokens["refresh_token"]
httpd.shutdown()
```

## 4. List agents

```python
from xct_litellm import XctClient

xct = XctClient(
    base_url="https://api.xct.test",
    access_token=tokens["access_token"],
    app_id="xct-agent-desktop",
)
agents = xct.agents.list(q="research", limit=20)
for a in agents:
    print(a["agent_id"], a["agent_name"], a.get("description"))
```

## 5. Stream an agent

```python
stream = xct.agents.invoke(
    agent_id="agent-rosalind",
    message={
        "role": "user",
        "parts": [{"text": "summarize the latest arxiv on LLM agents"}],
    },
    stream=True,
)
for event in stream:
    # SSE events parsed to dicts already
    if "content" in event:
        print(event["content"], end="", flush=True)
```

`stream=True` sets `Accept: text/event-stream`; server returns SSE; SDK
emits parsed dicts. NDJSON fallback is automatic if the proxy doesn't
honor SSE for that particular agent.

## 6. (Optional) Refresh tokens before they expire

```python
import time
from xct_litellm.pkce import PKCESession  # holds the helper

ACCESS_TOKEN_TTL = 3600  # see /oauth/token's expires_in
issued_at = time.time()

def access_token() -> str:
    nonlocal tokens, issued_at
    if time.time() - issued_at > ACCESS_TOKEN_TTL - 60:
        # use the refresh grant
        import httpx
        r = httpx.post("https://api.xct.test/oauth/token", data={
            "grant_type": "refresh_token",
            "client_id": "xct_abc...",
            "refresh_token": tokens["refresh_token"],
        })
        r.raise_for_status()
        tokens = r.json()
        issued_at = time.time()
    return tokens["access_token"]
```

Refresh tokens last 30 days; access tokens last 1 hour. Plan accordingly.

## Why a native quickstart looks different

Native apps can't redirect to a registered URL like a web app does. Two
options exist:
1. **Loopback** (this guide) — RFC 8252 recommended. Works offline.
2. **Custom URL scheme** (`xct-agent-desktop://oauth/callback`) — requires
   OS registration, more setup.

Loopback is cleaner because it requires zero OS plumbing and the proxy
already enforces exact-match `redirect_uri` whitelisting.

## Troubleshooting

- **403 on `/oauth/authorize`** → `redirect_uri` doesn't match the
  registered whitelist exactly (port included).
- **"unknown client_id"** → app is `is_active=false` in the DB or the
  client_id has a typo.
