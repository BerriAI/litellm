# LiteLLM Rust

This workspace contains the staged Rust implementation for LiteLLM.

`litellm-core` is the LiteLLM SDK in Rust: one entrypoint per top-level call
that makes the LLM call and hands back a typed response, the same shape as
`litellm.messages()` in Python.

```rust
let response = litellm_core::messages::messages(
    MessagesRequest {
        model: "claude-sonnet-4-5",
        body,
        options: RequestOptions {
            api_key: Some(key.to_string()),
            ..Default::default()
        },
    },
    &LiteLlmRequestContext::default(),
)
.await?;
```

Python continues to own configuration, retries, routing policy, logging,
callbacks, spend tracking, and customer plugins until each Rust path has parity
coverage and production evidence.

## Native request boundary

Native HTTP routes accept `native(request, *, context, callback_adapter=None, auth_provider=None)`
Responses WebSocket connections retain `connect(request, *, context, callback_adapter=None)`
The request carries the endpoint payload and `NativeRequestOptions`: credentials,
provider routing, headers, query parameters, and timeout. `NativeRequestContext`
carries LiteLLM metadata, call identity, and attribution separately from the provider payload

Python builds the frozen request dataclasses in `litellm/rust_bridge/request.py` and
PyO3 extracts their fields before execution. Provider connection parameters, such as
AWS credentials and Vertex project/location, belong in `options.provider_connection`
rather than the request body

The prepared call keeps `callback_adapter` and executable `auth_provider` separate from request data and execution
context. Explicit Python token providers decline before execution until their compatibility layer is implemented. HTTP callbacks use `OneShotCallbackHandle`; WebSocket callbacks use
`SessionCallbackHandle`. Construction happens after enablement and readiness checks

OCR exercises the callback foundation. No native route advertises callback readiness
yet, so normal SDK calls use Python where fallback exists. Required-native
transcription reports unavailable. Tests enable diagnostic bindings only in isolated
scopes. Existing chat and Messages provider preparation and logging remain in place

## Crates

| Crate | Role |
|-------|------|
| litellm-core | The SDK. Per-route entrypoints (`messages::messages()`), types, provider transforms (modules under `providers/`), provider resolution, auth, the provider HTTP call, and the router. |
| litellm-config | Config-loading boundary. Returns resolved deployments and optionally delegates loading to Python. |
| litellm-ai-gateway | The axum server (behind the `server` feature) and WebSocket hosts. Translates HTTP/WS to core entrypoints; no provider handlers. |
| litellm-python-interop | Domain-neutral PyO3 foundation for GIL handling and typed Python/Serde conversion. |
| litellm-python-bridge | PyO3 cdylib exposing LiteLLM Rust APIs to the Python SDK. Owns API registration, domain wiring, and Python exception mapping. |

Dependency direction is acyclic: config depends on core, the gateway depends on config and core, and the Python bridge depends on the domain layers and Python interop.

## Layout

```text
crates/
  core/           The SDK: route modules + provider transforms.
    src/messages/   mod.rs (entrypoint), types, transformation, prepare, handler, client
    src/providers/anthropic/messages/transformation.rs
  config/         Config loading and resolved deployments.
  ai-gateway/     Axum server + WebSocket hosts; calls core entrypoints.
  python-interop/ Domain-neutral PyO3 conversion and GIL primitives.
  python-bridge/  PyO3 API adapter for Python LiteLLM.
```

The folder shape follows the Python provider tree:
`core/src/providers/<provider>/<route>/transformation.rs`. The bridge exposes one
function per top-level route, mirroring the core entrypoints.

## Checks

Run the commands under "Checks" in [CLAUDE.md](CLAUDE.md) before pushing Rust
changes. That list is the single source of truth and matches what GitHub Actions
runs for changes under `litellm-rust/`.

## Shared provider authentication

`core::auth` separates typed adapter construction (`AuthAdapter`) from authorization of a final HTTP request (`RequestAuthorizer`). Bearer methods can compose `TokenProvider` with `BearerTokenAuthorizer`; header and signing methods use the request authorizer directly

An adapter produces an `AuthBinding`, which must be bound to a destination before use. `AuthHttpClient` validates the destination before invoking credentials, disables automatic redirects and bounds credential acquisition and HTTP send with the call deadline. A caller authorizes each permitted follow-up request again. Token providers distinguish known expiration from no-store tokens; the credential adapter owns refresh and reuse

The shared provider-attempt handler applies request replacements before serialization and authorization. The config layer constructs reusable native resources; the Python extension owns their lifetime. Calls use the runtime belonging to their defining extension module, including when wheels are loaded independently. The internal Python boundary carries an optional `auth_provider` separately from request data and lifecycle callbacks. It currently declines that unsupported mode before execution without invoking the callable

This foundation does not enable SDK routes or install native Azure/Google credential adapters. Consumer and method PRs must prove route, callback and authentication parity before readiness changes. Gateway authentication and custom gateway secret resolution are outside this SDK boundary
