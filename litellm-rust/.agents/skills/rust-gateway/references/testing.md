# Testing

Test HTTP contracts in the crate that owns the endpoint: status codes, forwarded headers, error envelopes and streaming behavior. Inference and provider behavior is tested in the `inference-<fmt>` and `llms` crates, and mounting and server lifecycle in `crates/gateway`

`crates/gateway-inference/tests/support/mod.rs` builds a `Router` with injected deployments, pools and settings through `app()` and its variants, and `post()` drives it with `tower::ServiceExt::oneshot`. `json()` reads the response body back. Use these instead of building a router or a client by hand in a test

Assert the status and the fields the surface promises, not the whole body. A test that pins a whole envelope breaks on every unrelated field, one that checks `status`, `error.type` and the presence of `request_id` catches the regressions that matter

`status()` and `body()` of the error type are pure, so a classification case belongs in the error module's unit tests, see the [`rust-error`](../../rust-error/references/gateway.md) skill. Router tests cover what only the full path shows, such as an extractor rejection arriving in the right envelope or an SSE error frame after the headers went out
