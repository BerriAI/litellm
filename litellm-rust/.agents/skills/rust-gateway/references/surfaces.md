# Envelopes follow the API surface

Each surface mirrors what the Python proxy returns there, so the shapes differ on purpose

| Surface | Envelope | Where |
|---|---|---|
| `/v1/messages` | Anthropic, `{"type": "error", "error": {"type", "message"}, "request_id"}` | `Error::body()` in `crates/gateway-inference/src/error.rs` |
| OpenAI-shaped routes | `{"error": {"message", "type", "param", "code"}}` | `Error::openai_response()` in the same file |
| MCP | `{"detail": message}` | `crates/gateway-mcp/src/error.rs` |
| UI | status plus a message | `crates/gateway-ui/src/error.rs` |
| auth rejections | plain text | `crates/gateway-auth/src/error.rs` |

Two rules hold on every surface. The error `type` string is derived from the status by `error_type()`, so a new failure never names its own type. A provider body already in the surface's shape passes through untouched, anything else has its message extracted and re-enveloped, and the body was bounded by `truncate_error_body` when the error was built

`/v1/messages` adds the request id to the envelope when the provider did not set one, which is why its handler takes the body as `Result<JsonObject, Error>` and renders the rejection itself, see [requests](requests.md)

A new surface gets its own renderer next to the others, named after the shape it produces, and its status still comes from `Error::status()`. Picking the status is the [`rust-error`](../../rust-error/references/gateway.md) skill's concern, not the surface's
