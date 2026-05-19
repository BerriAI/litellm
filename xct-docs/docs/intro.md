---
slug: /
sidebar_position: 1
---

# xct-litellm

The xct-litellm proxy turns the upstream LiteLLM proxy into a **capability
provider** for the XCT ecosystem (xct-chat, xct-home, xct-agent-desktop, …).

A consuming app talks to one unified surface and gets four entity classes:

| Class | What it is | Example |
|---|---|---|
| **Model** | An LLM behind a stable name | `deepseek-v3.2`, `gpt-4o` |
| **Agent** | A2A-protocol-compliant agent | A research agent, a writing assistant |
| **MCP** | Tool servers exposing capabilities via Model Context Protocol | A search MCP, a vector-store MCP |
| **Skill** | A prompt+tools bundle that augments chat completions | "fact-check", "summarize" |

## Read me in this order

1. **[Concepts → Overview](./concepts/overview.md)** — the four entity classes
   and how they relate.
2. **[Quickstart → xct-chat](./quickstart/xct-chat.md)** — concrete end-to-end
   integration in 30 minutes.
3. **[Recipes](./recipes/list-capabilities.md)** — short how-tos for the most
   common patterns.
4. **[API Reference](./api-reference/overview.md)** — auto-generated from
   `/openapi-public.json`.

## Where the code lives

| Component | Path |
|---|---|
| Proxy server | `litellm/proxy/` |
| Capability endpoints | `litellm/proxy/capability_endpoints/` |
| XCT-specific endpoints | `litellm/proxy/xct_app_endpoints/`, `xct_oauth_endpoints/`, `skill_endpoints/`, `webhook_endpoints/` |
| Python SDK | `sdk/python/` (`xct-litellm` on PyPI) |
| TypeScript SDK | `sdk/typescript/` (`@xct/litellm-sdk` on npm) |
| Dashboard | `ui/litellm-dashboard/` |
| These docs | `xct-docs/` |

## License

Apache-2.0, same as upstream LiteLLM.
