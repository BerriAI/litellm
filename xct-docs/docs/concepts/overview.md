---
sidebar_position: 1
---

# Overview

xct-litellm exposes four **capability classes** that downstream apps consume
through one auth model, one discovery endpoint, and one set of SDKs.

```
┌──────────────────────────────────────────────────────────────────┐
│                       xct-litellm proxy                          │
│                                                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                  │
│  │ Models │  │ Agents │  │  MCPs  │  │ Skills │   ◄── 4 classes  │
│  └────────┘  └────────┘  └────────┘  └────────┘                  │
│       │           │          │           │                        │
│       └──────┬────┴──────────┴───────────┘                        │
│              ▼                                                    │
│   GET /v1/capabilities  ◄── one discovery endpoint                │
│                                                                   │
│   POST /v1/chat/completions   (OpenAI-shape, plus `skills: [...]`)│
│   POST /v1/a2a/{id}/message/send                                  │
│   POST /mcp-rest/tools/call                                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
        ▲                ▲              ▲
        │                │              │
   xct-chat         xct-home     xct-agent-desktop
```

## Why "capability"

Each class is a **callable thing** with discoverable metadata, scoped to the
caller's tenancy. You don't ship hard-coded model lists or agent endpoints
in your app; you ask the proxy what's available and dispatch.

## The 4 classes in one paragraph

- **Model** — a stable name (`deepseek-v3.2`) that the proxy routes to one
  or more provider deployments. Standard OpenAI shape on the wire.
- **Agent** — an A2A-protocol-compliant agent. Has an agent card, a URL,
  optional streaming. Invoked as JSON-RPC 2.0.
- **MCP** — a Model Context Protocol server, exposing a set of tools. The
  proxy proxies tool calls and tracks spend per tool.
- **Skill** — a prompt template + optional tool schema, stamped into a chat
  completion request via `skills: ["fact-check@v3"]`. Server-side.

## How they bind together

The proxy can also bundle them. A `chat.completions.create()` call can:

1. Use a **model** (always)
2. Be augmented by one or more **skills** (system-prompt + tool injection)
3. Include **MCP tools** in the `tools` array
4. (Streaming) yield SSE or NDJSON chunks the same way `agents.invoke()` does

Everything you can discover via `GET /v1/capabilities`, you can call.

## Tenancy

See **[App-Tenancy](./app-tenancy.md)** for the per-app filtering layer.
Short version: when a token carries an `app_id`, the capability response
narrows to that app's `capability_scope_id`, and every spend log / metric /
webhook is attributed to that app.
