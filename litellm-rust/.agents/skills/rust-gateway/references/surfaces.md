# Envelopes follow the API surface

Each surface mirrors what the Python proxy returns there, so the shapes differ on purpose

| Surface | Envelope | Where |
|---|---|---|
| `/v1/messages` | Anthropic, `{"type": "error", "error": {"type", "message"}, "request_id"}` | `Error::body()` and `sse_frame()` in `crates/gateway-inference/src/error.rs` |
| OpenAI-shaped routes | `{"error": {"message", "type", "param", "code"}}` | `Error::openai_response()` in the same file |
| `/v1/responses` stream | the Responses `error` event, `{"type": "error", "code", "message", "param"}` | `Error::responses_sse_frame()` in the same file |
| MCP | `{"detail": message}` | `crates/gateway-mcp/src/error.rs` |
| UI | `{"error": {"message"}}`, plus `Retry-After` on 429 | `crates/gateway-ui/src/error.rs` |
| auth rejections | plain text, the top-level message only | `crates/gateway-auth/src/error.rs` |

Every surface takes its status from one `status()` on its crate's `Error` and renders the message of anything that is not a provider answer with `litellm_error::Report`. In `gateway-inference` the error `type` string is derived from the status by `error_type()`, so a new failure never names its own type. The Anthropic envelope passes a provider body already in its shape through untouched and extracts the message from anything else, and the OpenAI envelope always wraps, as the Python proxy does. A provider body is bounded by `litellm_http::Error::upstream` when the error is built

Two surfaces show less than the chain on purpose, and a test in each pins it. An auth rejection goes to a caller who has not authenticated, so it never shows what a secret manager or key store said. An MCP upstream failure shows only its own layer, because rmcp renders the transport error, URL included, into its own `Display`

`/v1/messages` adds the request id to the envelope when the provider did not set one, which is why its handler takes the body as `Result<JsonObject, Error>` and renders the rejection itself, see [requests](requests.md)

A new surface gets its own renderer next to the others, named after the shape it produces, and its status still comes from `Error::status()`. Picking the status is the [`rust-error`](../../rust-error/references/gateway.md) skill's concern, not the surface's
