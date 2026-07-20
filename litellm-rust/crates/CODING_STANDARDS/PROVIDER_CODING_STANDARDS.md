# Provider coding standards (litellm-rust)

Rules for adding or changing an LLM provider/route in `litellm-rust`. OCR (`MISTRAL_OCR_CONFIG`) is the reference; `messages` (`ANTHROPIC_MESSAGES_CONFIG`) is the next port.

## Provider resolution

1. Always resolve the provider/model first with `get_custom_llm_provider` (`core/src/routing_utils/provider.rs`). Nothing downstream may branch on a raw model string.
2. Model/provider is resolved once, in `prepare.rs`, and passed down as typed fields. Don't re-resolve or re-parse it in transforms or handlers.

## Transforms and the base config

3. Every route defines a base config trait with `transform_request` + `transform_response` (+ `complete_url`, `supported_params`), living in `core/src/<route>/transformation.rs` (e.g. `AnthropicMessagesProviderConfig`, mirroring `OcrProviderConfig`).
4. Each provider implements that trait as a `const <PROVIDER>_<ROUTE>_CONFIG` in `core/src/providers/<provider>/<route>/transformation.rs`, mirroring the Python provider tree.
5. Individual configs implement only the request/response transforms. Shared behavior (param filtering, defaults) stays as trait default methods so future providers inherit existing logic instead of reimplementing it.
6. Prefer composition: a provider that extends another reuses the base trait's defaults or wraps another config; don't copy transform bodies between providers.

## Boundaries

7. Layers never cross: `core` = pure transforms/types (no network, env, secrets, auth, logging, global mutable state); `ai-gateway` = all I/O, auth headers, HTTP/SSE, lifecycle hooks; `python-bridge` = thin PyO3 adapter.
8. Generic/route files contain zero provider-specific branches. A provider is one module under `core/src/providers/<provider>/<route>/`; a route is a module, never a new crate.
9. Route entry point stays thin: `<route>()` -> `prepare_*` -> `CallLifecycle::run_request`, which owns the pre_call -> during_call -> provider call -> success/failure order and phase timing. Handlers validate and delegate; no business logic in them.
10. Constants (URLs, env-var names, API versions, error messages) live in a crate `constants.rs`, never inline. Env reads happen only at the host/config layer, with the `DEFAULT_*` fallback defined in `constants.rs`.

## Types and errors

11. Typed contracts only: no bare `serde_json::Value` / `String` / `Vec<String>` as a transform input or output. Parse wire bytes into typed structs/enums at the host edge; a `type` discriminator is a typed field, not a raw string.
12. Model failures as values: return typed `CoreError`, don't panic. No `unwrap`/`expect`/`panic!` on user or provider input.
13. No mutation: build values in one shot (comprehensions/iterators, `collect`), prefer immutable bindings and owned typed structs over seeding-and-mutating.
14. Early returns over deep nesting; small focused files over god modules.
15. Preserve Python output shape intentionally. If a field is always serialized as `null` for parity, keep it and pin it with a test.

## Safety and data minimization

16. Never log request/response bodies, base64 payloads, document contents, or secrets. Truncate and bound any upstream body before it crosses a host boundary.
17. Treat empty/whitespace credentials, URLs, and config values as absent at the host resolution layer.
18. Host I/O sets connect + request timeouts (no unbounded waits), reuses a shared HTTP client, and prefers rustls TLS.

## Tests and rollout

19. Every provider transform ships tests for: supported-param filtering, request body shape, response normalization, missing/null fields, bad input, and `*_match_python` fixture parity.
20. Lifecycle/hook tests cover hook order, success + failure callback payloads, pre-call guardrail blocking before any provider I/O, during-call body mutation, and provider-error mapping.
21. When a route has a Python reference implementation, the Rust path stays off by default and behind Python parity tests (disabled / enabled-equals-Python / bridge-unavailable fallback) until parity is proven. A new provider/route may instead be implemented rust-only with no Python reference; then the Python interface is a thin dispatch to Rust with no fallback, and tests cover the rust-backed path plus the unavailable-bridge error. State the rust-only choice explicitly in the PR.

## Python bridge (SDK side)

22. A Python -> Rust bridge keeps the Python side minimal: the Python interface only marshals inputs and calls the Rust interface, with no transform, handler, or business logic. Aim for well under 100 lines of interface code per route; if the Python grows past that, the logic belongs in Rust.
23. Do not bloat `litellm/main.py`. A route's provider dispatch lives in a thin dispatch class under `litellm/llms/<provider>/<route>/` that calls the Rust bridge; `main.py` only instantiates it and calls its sync/async method.
24. Do not add new feature flags unless explicitly requested. Reuse the existing litellm rust rollout mechanism (`use_litellm_rust`); never introduce a per-route env flag such as `LITELLM_USE_RUST_<ROUTE>`.

## Checks before push

25. Run, and keep green:
    ```bash
    cd litellm-rust
    cargo fmt --check
    cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings
    cargo clippy -p litellm-core -p litellm-python-bridge --all-targets -- -D warnings
    cargo test --workspace
    ```
