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

Each hop is one `From` that nests the lower error in one variant, see [conversions](conversions.md). No hop re-derives what the lower crate already decided, and nothing between the provider and `IntoResponse` formats an error for the client. That is what makes "one `Error` with `IntoResponse` per gateway crate that serves a router" work: a handler can `?` any failure from any depth and the response comes out right

## Classify once

`status()` is pure and runs once per error. Variants the gateway declares itself, such as a bad body, an unknown model or an oversized body, are matched at the top of its own enum. A nested route error goes to `route_status` and an OCR error to `ocr_status`, which ask the predicates [boundaries](boundaries.md) lists

| Answer | Status |
|---|---|
| `declined()` | 501, for a route error, the gateway matched and parsed the request but core cannot serve it, nothing reached the provider |
| `upstream_response()` | the provider's own status, 502 when it is not a valid code |
| `is_missing_credential()` | 401 |
| `is_request()` | 400, for a route error |
| `http_status_code()` | the transport's status, for an OCR error, such as 408 for a timeout |
| otherwise | 500 |

Do not reach into a nested enum's variants from the gateway. When a new nested failure needs a different status, add a predicate on the nested error, see [classification](classification.md)

## Render by surface and framing

Rendering is separate from classification. The surface picks the envelope and the moment picks the framing, a JSON body before the headers went out and an `event: error` SSE frame after. Both framings of one surface render the same status and `Report` message, so a streaming failure and a JSON failure of the same error agree on everything but framing

| Surface | JSON | SSE |
|---|---|---|
| Anthropic, `/v1/messages` | `body(request_id)` | `sse_frame()` |
| OpenAI, chat, responses and the rest | `openai_response()` | `responses_sse_frame()`, the Responses API `error` event |

The message of anything that is not a provider answer is the `litellm_error::Report` of the error, and in the OpenAI and Anthropic JSON envelopes the error `type` string is derived from the status by `error_type()`, so a new failure never names its own type. The Responses `error` event has the fixed type `error` and the status as its `code`. The Anthropic envelope passes a provider body already in its shape through untouched and extracts the message from anything else. The OpenAI envelope always wraps, because that is what the Python proxy returns there

## Every way in uses the same Error

A request can fail before the handler runs, in the handler, or after the headers went out. All three end in the same `Error`, so they get the same classification and envelope

- An extractor sets `type Rejection = Error`, so a bad body is the surface's envelope rather than axum's plain text
- A path the gateway knows but does not serve returns `Error::Unsupported(path)` from one handler, so the client gets a 501 in the surface's envelope rather than a bare 404
- A stream that fails after its headers went out hands the surface's SSE frame of `Error::from(error)` to `litellm_host_http::Sse::new`. `litellm_host_http::Error<E>` separates `Call(E)`, the route's own error, from `Protocol`, a host bug, and the `From` maps the second to `Internal`

The axum hooks behind each line are in [`rust-gateway`](../../rust-gateway/SKILL.md)

## Testing

`status()` and `body()` are pure, so unit tests call them on errors built the way production builds them, one `#[case]` per fault, as `status_follows_who_is_at_fault`, `an_anthropic_provider_body_passes_through` and `other_failures_get_the_anthropic_envelope_with_their_message` in `crates/gateway-inference/src/error.rs` do. Assert the status and the fields the surface promises, not the whole body, except where passing the body through untouched is the contract. Driving a whole router belongs to the gateway crate's own tests, see [`rust-gateway`](../../rust-gateway/references/testing.md)
