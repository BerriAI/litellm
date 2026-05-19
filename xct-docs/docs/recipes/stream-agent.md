---
sidebar_position: 2
---

# Stream an agent invocation

A2A `message/stream` over SSE.

```python
stream = xct.agents.invoke(
    agent_id="agent-rosalind",
    message={"role": "user", "parts": [{"text": "..."}]},
    stream=True,
)
for event in stream:
    # event is the parsed JSON payload of one SSE frame
    print(event)
```

TypeScript equivalent:
```ts
for await (const event of xct.agents.invoke("agent-rosalind", {
  message: { role: "user", parts: [{ text: "..." }] },
  stream: true,
})) {
  console.log(event);
}
```

SDK sets `Accept: text/event-stream`. Server emits frames like:
```
event: a2a.message
data: {"role":"assistant","content":"Hello"}

event: a2a.error
data: {"jsonrpc":"2.0","id":"req-1","error":{"code":-32603,"message":"..."}}
```

NDJSON is the fallback wire format — the SDK auto-detects and parses
both. You don't have to choose.

**Client disconnect** cancels the upstream agent task — FastAPI's
StreamingResponse handles that. No leak.
