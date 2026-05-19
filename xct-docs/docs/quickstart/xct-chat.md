---
sidebar_position: 1
---

# Quickstart: xct-chat

Get an xct-chat-style web app talking to the proxy in **30 minutes**.

## What you'll build

- A React page that PKCE-logs the user into the proxy
- After login, the page shows the user's available **capabilities**
- A simple "chat" form sends `/v1/chat/completions` requests
- Streaming responses render as they arrive

## 1. Provision the app (admin)

In the dashboard → **XCT Apps** → **New app**:

| Field | Value |
|---|---|
| `app_name` | `xct-chat` |
| `display_name` | `XCT Chat` |
| `redirect_uris` | `http://localhost:3000/oauth/callback` (for dev) |
| `default_team_id` | (your team UUID) |
| `default_scopes` | `read write` |
| `capability_scope_id` | (optional access group) |

Save the modal's one-time `client_secret`. The `oauth_client_id` is stable
and lives in the row.

## 2. Install the SDK

```bash
npm install @xct/litellm-sdk
```

## 3. Wire up the PKCE flow

`oauth/start.tsx`:
```tsx
import { beginPkce } from "@xct/litellm-sdk";

export async function startLogin() {
  const session = await beginPkce({
    baseUrl: "https://api.xct.test",
    clientId: "xct_abc...",   // from step 1
    redirectUri: "http://localhost:3000/oauth/callback",
    scope: "read write",
  });
  window.location.href = session.authorizeUrl;
}
```

`oauth/callback.tsx`:
```tsx
import { completePkce } from "@xct/litellm-sdk";

export default async function Callback() {
  const url = new URL(window.location.href);
  const tokens = await completePkce("https://api.xct.test", {
    code: url.searchParams.get("code")!,
    state: url.searchParams.get("state") ?? undefined,
  });
  sessionStorage.setItem("xct_access_token", tokens.access_token);
  sessionStorage.setItem("xct_refresh_token", tokens.refresh_token!);
  window.location.href = "/";
}
```

Note: tokens live in **sessionStorage** only (CLAUDE.md convention — no
localStorage for credentials). The PKCE state is cleared automatically.

## 4. List capabilities + render

```tsx
import { XctClient } from "@xct/litellm-sdk";

const xct = new XctClient({
  baseUrl: "https://api.xct.test",
  accessToken: sessionStorage.getItem("xct_access_token")!,
  appId: "xct-chat",   // optional; token already carries it
});

export async function loadCapabilities() {
  const caps = await xct.capabilities.list();
  // caps.models[].id, caps.agents[].agent_id, caps.mcps[].server_id, …
  return caps;
}
```

The proxy already narrowed this list to what your app + user can use. No
client-side filtering needed.

## 5. Streaming chat

```tsx
async function sendMessage(model: string, messages: Array<{role: string, content: string}>) {
  for await (const chunk of xct.chat.completions.create({
    model,
    messages,
    stream: true,
  })) {
    const delta = (chunk as any).choices?.[0]?.delta?.content;
    if (delta) appendToView(delta);
  }
}
```

That's it. The proxy:
- Authenticated via OAuth (S4 flow)
- Stamped `app_id="xct-chat"` on the request (S4-04)
- Recorded the call in spend logs with `entity_type="model"` (S6-01)
- Fired `capability.invoked` to any webhook subscribers (S6-06)

## 6. (Optional) Inject a skill

```tsx
await xct.chat.completions.create({
  model: "deepseek-v3.2",
  messages: [{ role: "user", content: "Is this claim true: ..." }],
  // @ts-ignore — XCT-specific extension to OpenAI shape
  skills: ["fact-check@v3"],
});
```

The proxy injects `fact-check@v3`'s system prompt + tools before
calling the provider. Spend log row will have `entity_type="skill"`,
`entity_id="fact-check"`.

## Troubleshooting

- **`/v1/capabilities` returns empty** → see the
  [empty-capabilities recipe](../recipes/debug-empty-capabilities.md).
- **OAuth state mismatch** → sessionStorage was cleared mid-flow.
- **401 after token refresh** → call `refreshAccessToken()` and update
  `xct.accessToken`.
