# Requests

## Extractors reject with the crate's Error

A custom extractor sets `type Rejection = Error`, so a bad body gets the surface's envelope rather than axum's plain-text rejection. `JsonObject` and `InferenceBody` in `crates/gateway-inference/src/request.rs` are the pattern

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

When the handler needs something from the request in the error body, such as the request id on `/v1/messages`, take the extractor as `Result<JsonObject, Error>` so the handler maps the rejection itself, as `messages::create` does. `Infallible` is the right rejection for an extractor that only reads an optional header, as `RequestId` does

## Paths

Every alias of a route is registered together in `crates/gateway-inference/src/lib.rs`, the bare path, its `/v1` form and the deployment-prefixed form, otherwise one alias silently behaves differently from the others. A path the gateway knows but does not serve, such as `/v1/completions`, is routed to `request::unsupported`, which returns `Error::Unsupported(path)` so the client gets a 501 in the surface's envelope rather than a bare 404

`crates/gateway` merges the mounted routers and adds the shared inbound middleware as `route_layer`s, so authentication runs only for matched routes. Endpoint paths and parsing stay in the mounted crate, see the two crates' `AGENTS.md`

## Streaming

Once the response headers are sent the status cannot change. A streaming handler builds `litellm_host_http::Sse::new(Json, |error| Bytes::from(Error::from(error).sse_frame()))`, where the closure renders a mid-stream failure as an `event: error` frame through the same `body()` a JSON failure would use. `litellm_host_http::Error<E>` separates `Call(E)`, the route's own error, from `Protocol`, a host bug, and the crate's `From` maps the second to `Internal`
