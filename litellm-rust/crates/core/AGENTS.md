litellm-core owns route orchestration. Messages and HTTP Responses return `litellm_host::call::CallOutput`, containing either a completed response or a stream head and chunks. OCR and currently non-streaming Chat Completions return their completed response directly

Each route exposes an ordinary async entrypoint and a `*_with_hooks` variant. Both use the same preparation and handler functions. Handlers accept `RouteHooks`, never a concrete `ChannelHooks`. Native observers receive start and terminal events through the shared call runner; a stream retains its lifecycle until exhaustion, error, or drop. A host channel has no native observer because its driver owns terminal dispatch

`route.rs` declares the concrete `Protocol` and prepares a factory accepting a typed request and constructing a `litellm_host::call::HostedMachine` with `hosted_call`. The shared call plumbing owns stream opening, delivery, backpressure, and detachment. Request decoding belongs to the boundary before the machine starts. Route closures only supply execution dependencies and route-specific host capabilities such as an OCR token provider. Use `run_hosted` for a native host so detachment is reported as cancellation. Python uses its own shared driver and preserves caller-task callback execution

Responses WebSocket sessions remain separate from the HTTP call driver because a connection can accept multiple requests while receiving events

## Crate layering

Each crate mirrors one top-level Python package, so a Rust path reads as its Python path with the crate name in place of the package directory. Dependencies only point down:

- `litellm-types` mirrors `litellm/types/`: pure serde data, no I/O
- `litellm-core-utils` mirrors `litellm/litellm_core_utils/`: pure helpers (provider resolution, prompt factory, call arguments, settings lookup and layer merge), no network I/O
- `litellm-http` is Rust-only and route-neutral: settings resolution, the pooled `reqwest` clients, TLS, proxies, the SSRF-safe media fetcher, request and header helpers, and transport errors. Python's `litellm/llms/custom_httpx/` is split by responsibility instead of mirrored: its transport half lives here, its OCR handler in `litellm-llms`
- `litellm-llms` mirrors `litellm/llms/`: `base_llm/<api>/transformation.rs`, `<provider>/<api>/transformation.rs`, and `base_llm/ocr/handler.rs` (the OCR request handler)
- `litellm-core` mirrors the route packages (`litellm/ocr/`, `litellm/messages/`, ...): entrypoints, route request types, provider dispatch, the route machine, and hooks

A route module owns the call entrypoint, route request types (`*Request<'a>`), credential fallback, provider dispatch, and the handler glue that runs a provider config. Provider code never imports from core; when it needs the caller's hooks mid-call it goes through `litellm_llms::base_llm::ocr::handler::CallHooks`, the provider-level hooks OCR implements over its host until it folds into `litellm_host::hooks::RouteHooks`. Import every item from its canonical path. Never re-export another crate's items or give an item a second public path; the only re-export allowed is a private submodule surfacing its item at its module root (`mod error; pub use error::Error;`). Handlers belong in core or llms, never in a host crate

## Error placement

The workspace `Error definitions` rules shape each crate's error; this section decides which crate and module a failure belongs to

A failure is declared once, by the lowest crate that raises it. Every crate above nests that error unchanged (`#[error(transparent)] Auth(#[from] litellm_auth::Error)`) or maps it once at its boundary, as `src/error.rs` does for `litellm_llms::Error`. `RouteError` collects route failures and never re-declares a variant a lower crate raises

Scope follows the concept, not the first caller. An error type under `litellm-llms`'s `<provider>/` is private to that provider: no other provider and nothing in `base_llm` may import it. A failure two providers or two routes can hit, such as wire framing, stream event decoding, or a malformed provider response, belongs to the crate that owns the concept: `litellm-framing` for framing, `litellm_llms::Error` for the transformation layer

`litellm_llms::Error` (`crates/llms/src/error.rs`) is the one transformation error for every provider and API. `base_llm/ocr/error.rs` is the recorded exception until OCR folds into it

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or callback execution of any kind. Core runs each route as a machine that yields host operations and call events; which integrations consume those events is the host's business.
