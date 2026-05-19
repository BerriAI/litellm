---
sidebar_position: 1
---

# API Reference

The full API reference is **generated from the live proxy's
`/openapi-public.json`** — never hand-maintained. The generator is at
`xct-docs/scripts/generate-api-reference.mjs` and CI runs it nightly.

## Surfaces

| Group | Routes |
|---|---|
| **Discovery** | `GET /v1/capabilities`, `GET /.well-known/xct-capabilities`, `GET /v1/models` |
| **Chat** | `POST /v1/chat/completions`, `POST /v1/completions` |
| **Agents (read)** | `GET /v1/agents`, `GET /v1/agents/{id}`, `GET /v1/agents/{id}/.well-known/agent-card.json` |
| **A2A** | `POST /v1/a2a/{id}/message/send` (with `Accept: text/event-stream` for SSE) |
| **MCP** | `GET /v1/mcp/tools`, `POST /mcp-rest/tools/call`, URL-namespaced `/{server}/mcp/v1/chat/completions` |
| **Skills (read)** | `GET /v1/xct-skills`, `GET /v1/xct-skills/{id}` |
| **OAuth** | `GET /oauth/authorize`, `POST /oauth/token`, `POST /oauth/revoke`, `POST /oauth/introspect` |
| **Other** | `POST /v1/embeddings`, `POST /v1/responses`, `POST /v1/messages` (Anthropic pass-through) |

Admin / write surfaces (`/v1/xct-apps`, `/v1/webhooks`,
`POST /v1/xct-skills`, `POST /v1/agents`, etc.) are **not** in the
public OpenAPI by design. They're available via `/openapi.json` for
proxy admins.

## Fetch the spec yourself

```bash
curl https://api.xct.test/openapi-public.json > openapi-public.json
```

Drop it into your favorite OpenAPI viewer (Swagger UI, Redoc, Stoplight),
generate client code with `openapi-generator-cli`, etc.

## Already wrapped by SDKs

- TS: `@xct/litellm-sdk` ([source](https://github.com/XcityUS/xcity-litellm/tree/litellm_internal_staging/sdk/typescript))
- Python: `xct-litellm` ([source](https://github.com/XcityUS/xcity-litellm/tree/litellm_internal_staging/sdk/python))

If you can use one of these, you don't need to touch OpenAPI at all.
