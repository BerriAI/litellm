# @xct/litellm-sdk

TypeScript SDK for the xct-litellm capability provider.

```ts
import { XctClient, beginPkce, completePkce } from "@xct/litellm-sdk";

// --- server-side: existing access_token ---
const xct = new XctClient({
  baseUrl: "https://api.xct.test",
  accessToken: "sk-xct-...",
  appId: "xct-chat",
});

// Discovery
const caps = await xct.capabilities.list();

// Chat (OpenAI-shape)
const reply = await xct.chat.completions.create({
  model: "deepseek-v3.2",
  messages: [{ role: "user", content: "hi" }],
});

// Streaming via SSE
for await (const chunk of xct.chat.completions.create({
  model: "deepseek-v3.2",
  messages: [{ role: "user", content: "hi" }],
  stream: true,
})) {
  console.log(chunk);
}

// A2A agent invocation
const stream = xct.agents.invoke("agent_rosalind", {
  message: { role: "user", parts: [{ text: "..." }] },
  stream: true,
});
for await (const event of stream) console.log(event);

// MCP tools available to this caller
const tools = await xct.mcp.listTools();

// --- browser-side: PKCE login flow ---
const session = await beginPkce({
  baseUrl: "https://api.xct.test",
  clientId: "xct_abc",
  redirectUri: "https://chat.xct.test/oauth/callback",
});
window.location.href = session.authorizeUrl;

// at the redirect_uri, after user returns:
const url = new URL(window.location.href);
const tokens = await completePkce("https://api.xct.test", {
  code: url.searchParams.get("code")!,
  state: url.searchParams.get("state") ?? undefined,
});
// tokens.access_token + tokens.refresh_token
```

## Surface

| Resource | Methods |
|---|---|
| `xct.capabilities` | `list()` |
| `xct.agents` | `list({q, category, tag, cursor, limit})`, `get(id)`, `invoke(id, {message, stream?})` |
| `xct.mcp` | `listTools()` |
| `xct.skills` | `list({q, team_id, cursor, limit})`, `get(id)` |
| `xct.chat.completions` | `create({...openai-shape, stream?})` |
| `beginPkce`, `completePkce`, `refreshAccessToken` | OAuth helpers |

## Streaming

`chat.completions.create({stream: true})` and `agents.invoke({stream: true})`
return `AsyncIterable<T>`. Server responds with **text/event-stream** when the
SDK sends `Accept: text/event-stream` (which it does for stream:true), with
NDJSON as a fallback for old A2A clients. The parser handles both.

## Errors

All non-2xx responses throw an `XctError` subclass:

- 401 / 403 → `AuthError`
- 404 → `CapabilityNotFoundError`
- 429 → `RateLimitError`
- everything else → `XctError`

Each carries `.status` and `.body` for debugging.

## Storage convention

`PKCESession` state lives in `sessionStorage` only (NOT localStorage). The
proxy's project convention is the same — keys / OAuth tokens never go
through localStorage.
