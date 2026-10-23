# Gateway HTTP

Each gateway crate has one `Error` with `IntoResponse`, so a handler returns `Result<_, Error>` and every `?` ends in the right status and envelope. `crates/gateway-inference/src/error.rs` is the reference. Axum's own conventions are in its [error handling docs](https://docs.rs/axum/latest/axum/error_handling/index.html)

## Status and body are separate

`status()` picks the code and is pure. `body(request_id)` builds the Anthropic envelope, `openai_response()` the OpenAI one, and `sse_frame()` reuses `body()` for a stream that fails after its headers went out. Keeping them apart is what lets a streaming failure and a JSON failure of the same error agree on everything but framing

```rust
pub fn status(&self) -> StatusCode {
    match self {
        Self::Auth(error) => error.status(),
        Self::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
        Self::BodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        Self::InvalidBody(_) | Self::UnknownModel(_) => StatusCode::BAD_REQUEST,
        Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        Self::Ocr(error) => ocr_status(error),
        Self::Route(error) => route_status(error),
    }
}
```

Variants the gateway declares itself are matched. A nested crate's error goes to one function that asks its predicates in order: unsupported, then the provider's own status, then a missing credential, then the caller's fault, else 500. Do not reach into the nested enum's variants from the gateway, add a predicate on the nested error instead, see [classification](classification.md)

## Envelopes follow the API surface

Each surface mirrors what the Python proxy returns there, so the shapes differ on purpose

| Surface | Envelope |
|---|---|
| `/v1/messages` | Anthropic, `{"type": "error", "error": {"type", "message"}, "request_id"}` from `body()` |
| OpenAI-shaped routes | `{"error": {"message", "type", "param", "code"}}` from `openai_response()` |
| MCP | `{"detail": message}` in `crates/gateway-mcp` |
| UI | `{"error": {"message"}}` in `crates/gateway-ui` |
| auth rejections | plain text in `crates/gateway-auth` |

The error `type` string is derived from the status, so a new failure never names its own type. A provider body already in the surface's shape passes through untouched, anything else has its message extracted and re-enveloped. The body was bounded by `truncate_error_body` when the error was built, see [boundaries](boundaries.md)

## Extractors reject with the same Error

A custom extractor sets `type Rejection = Error`, so a bad body gets the surface's envelope rather than axum's plain-text rejection

```rust
impl<S: Send + Sync> FromRequest<S> for JsonObject {
    type Rejection = Error;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let body = Bytes::from_request(request, state).await.map_err(|error| {
            if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
                Error::BodyTooLarge
            } else {
                Error::InvalidBody(error.to_string())
            }
        })?;
        object(&body).map(Self)
    }
}
```

When the handler needs the request id in the error body, take the extractor as `Result<JsonObject, Error>` so the handler maps the rejection itself, as `messages::create` does. `Infallible` is the right rejection for an extractor that reads an optional header

A path the gateway knows but does not serve, such as `/v1/completions`, is routed to `request::unsupported`, which returns `Error::Unsupported(path)` so the client gets a 501 in the surface's envelope rather than axum's bare 404

## Streaming

Once the response headers are sent the status cannot change. `litellm_host_http::Sse::new(Json, |error| Bytes::from(Error::from(error).sse_frame()))` takes the closure that renders a mid-stream failure as an `event: error` frame. `litellm_host_http::Error<E>` separates `Call(E)`, the route's own error, from `Protocol`, a host bug, which the `From` maps to `Internal`

## Known gaps

`InvalidBody(String)` and `Internal(String)` flatten their cause into a string, against [messages and causes](messages-and-causes.md). A new variant carries the typed cause

## Testing

`status()` and `body()` are pure, so unit tests call them on errors built the way production builds them, one `#[case]` per fault, as `status_follows_who_is_at_fault` and `body_follows_the_anthropic_exception_mapping` do. For the full path, `tests/support/mod.rs` builds a `Router` and `post()` drives it with `tower::ServiceExt::oneshot`. Assert the status and the fields the surface promises, not the whole body
