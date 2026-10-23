# Gateway responses

The gateway is where a Rust error becomes an HTTP response. This page is about the path an error takes to get there and the one place it is classified. The gateway crates' own conventions, such as which envelope each API surface uses, which extractors exist and how the router is composed, live in the [`rust-gateway`](../../rust-gateway/SKILL.md) skill

## The flow

```text
reqwest / provider failure
  -> litellm_http::Error                 From<reqwest::Error>, strips the URL, classifies once
  -> litellm_inference::Error::Http(_)   From, one transparent variant, predicates delegate
  -> gateway_inference::Error::Route(_)  From, one variant per nested crate
  -> handler returns Result<_, Error>    every `?` ends here
  -> IntoResponse                        status() picks the code, a renderer builds the body
```

Each hop is one `From` that nests the lower error in one variant, see [conversions](conversions.md). No hop re-derives what the lower crate already decided, and nothing between the provider and `IntoResponse` formats an error for the client. That is what makes "one `Error` with `IntoResponse` per gateway crate" work: a handler can `?` any failure from any depth and the response comes out right

## Classify once

`status()` is pure and runs once per error. Variants the gateway declares itself, such as a bad body, an unknown model or an oversized body, are matched at the top of its own enum. A nested crate's error goes to one function that asks the predicates in the order [boundaries](boundaries.md) gives, which here means

| Answer | Status |
|---|---|
| `declined()` | 501, the gateway matched and parsed the request but core cannot serve it, nothing reached the provider |
| `upstream_response()` | the provider's own status, 502 when it is not a valid code |
| `is_missing_credential()` | 401 |
| `is_request()` | 400 |
| otherwise | 500 |

Do not reach into a nested enum's variants from the gateway. When a new nested failure needs a different status, add a predicate on the nested error, see [classification](classification.md)

## Render by surface and framing

Rendering is separate from classification. The surface picks the envelope, which is why `gateway_inference::Error` has both `body(request_id)` for the Anthropic shape and `openai_response()` for the OpenAI shape, and the moment picks the framing: a JSON body before the headers went out, an `event: error` SSE frame from `sse_frame()` after. Both framings render the same `body()`, so a streaming failure and a JSON failure of the same error agree on everything but framing

Two rules hold for every surface. The error `type` string is derived from the status, so a new failure never names its own type. A provider body already in the surface's shape passes through untouched and anything else has its message extracted, with the body bounded by `truncate_error_body` when the error was built, see [boundaries](boundaries.md)

## Every way in uses the same Error

A request can fail before the handler runs, in the handler, or after the headers went out. All three end in the same `Error`, so they get the same classification and envelope

- An extractor sets `type Rejection = Error`, so a bad body is the surface's envelope rather than axum's plain text
- A path the gateway knows but does not serve returns `Error::Unsupported(path)` from one handler, so the client gets a 501 in the surface's envelope rather than a bare 404
- A stream that fails after its headers went out hands `Error::from(error).sse_frame()` to `litellm_host_http::Sse::new`. `litellm_host_http::Error<E>` separates `Call(E)`, the route's own error, from `Protocol`, a host bug, and the `From` maps the second to `Internal`

The axum hooks behind each line are in [`rust-gateway`](../../rust-gateway/SKILL.md)

## Known gaps

`InvalidBody(String)` and `Internal(String)` flatten their cause into a string, against [messages and causes](messages-and-causes.md). A new variant carries the typed cause

## Testing

`status()` and `body()` are pure, so unit tests call them on errors built the way production builds them, one `#[case]` per fault, as `status_follows_who_is_at_fault` and `body_follows_the_anthropic_exception_mapping` in `crates/gateway-inference/src/error.rs` do. Assert the status and the fields the surface promises, not the whole body. Driving a whole router belongs to the gateway crate's own tests, see [`rust-gateway`](../../rust-gateway/references/testing.md)
