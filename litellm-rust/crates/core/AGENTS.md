litellm-core owns route orchestration. Messages and HTTP Responses return `litellm_host::call::CallOutput`, containing either a completed response or a stream head and chunks. OCR and currently non-streaming Chat Completions return their completed response directly

Hosts assemble route objects from shared `CoreResources`, HTTP settings, and secret sources. Each route owns its provider client and authentication dependencies. Gateway routes live for the gateway lifetime; Python assembles routes per call from its settings snapshot

Chat Completions, Messages, Responses, and OCR execute through their route objects. Calls pass `Interceptors` and an optional `ObservationSender` separately; use `&()` for no hooks and `None` for no observer. Construction does no work; preparation and lifecycle observation begin when the future is polled. Handlers accept `Interceptors`, never a concrete `ChannelInterceptors`. Native observers receive start and terminal events through the shared call runner; a stream retains its lifecycle until exhaustion, error, or drop. Hosted routes leave terminal observation to their driver

`route.rs` declares the concrete `Protocol` and implements a route method that accepts a typed request and constructs a `litellm_host::call::HostedMachine` with `hosted_call`. The shared call plumbing owns stream opening, delivery, backpressure, and detachment. Request decoding belongs to the boundary before the machine starts. Route closures only supply execution dependencies and route-specific host capabilities such as an OCR token provider. Use `run_hosted` for a native host so detachment is reported as cancellation. Python uses its own shared driver and preserves caller-task callback execution

Responses WebSocket sessions remain separate from the HTTP call driver because a connection can accept multiple requests while receiving events

## Crate layering

For Messages, Responses, Chat Completions, OCR, and other API formats, `core/src/<format>/` owns orchestration. Shared API data contracts belong in `litellm-llms-types`, adapter contracts and shared transformation machinery in `llms/src/base_llm/<format>/`, and provider policy in `llms/src/<provider>/<format>/`. A repeated format directory name does not imply interchangeable responsibilities. Select concrete adapters here, then invoke their contracts instead of applying one provider's policy to every call. Route types describe call envelopes and execution state, not duplicate public payload schemas

Crates separate API data, transformations, transport, and orchestration. Python package names identify counterparts, not ownership. Dependencies only point down:

- `litellm-llms-types` owns shared inference API contracts, grouped by format: pure serde data and shape validation, no I/O
- `litellm-core-utils` mirrors `litellm/litellm_core_utils/`: pure helpers (provider resolution, prompt factory, call arguments, settings lookup and layer merge), no network I/O
- `litellm-http` is Rust-only and route-neutral: settings resolution, the pooled `reqwest` clients, TLS, proxies, the SSRF-safe media fetcher, request and header helpers, and transport errors. Python's `litellm/llms/custom_httpx/` is split by responsibility instead of mirrored: its transport half lives here, its OCR handler in `litellm-llms`
- `litellm-llms` mirrors `litellm/llms/`: `base_llm/<api>/transformation.rs`, `<provider>/<api>/transformation.rs`, and `base_llm/ocr/handler.rs` (the OCR request handler)
- `litellm-core` mirrors the route packages (`litellm/ocr/`, `litellm/messages/`, ...): entrypoints, route request types, provider dispatch, the route machine, and hooks

A route module owns the call entrypoint, route request types (`*Request<'a>`), credential fallback, provider dispatch, and the handler glue that runs a provider config. Provider code never imports from core; when it needs the caller's hooks mid-call it goes through `litellm_llms::base_llm::ocr::handler::CallHooks`, the provider-level hooks OCR implements over its host until it folds into `litellm_host::interceptors::Interceptors`. Import every item from its canonical path. Never re-export another crate's items or give an item a second public path; the only re-export allowed is a private submodule surfacing its item at its module root (`mod error; pub use error::Error;`). Handlers belong in core or llms, never in a host crate

## Error placement

The workspace `Error definitions` rules shape each crate's error; this section decides which crate and module a failure belongs to

A failure is declared once, by the lowest crate that raises it. Every crate above nests that error unchanged (`#[error(transparent)] Auth(#[from] litellm_auth::Error)`) or maps it once at its boundary, as `src/error.rs` does for `litellm_llms::Error`. `RouteError` collects route failures and never re-declares a variant a lower crate raises

Scope follows the concept, not the first caller. An error type under `litellm-llms`'s `<provider>/` is private to that provider: no other provider and nothing in `base_llm` may import it. A failure two providers or two routes can hit, such as wire framing, stream event decoding, or a malformed provider response, belongs to the crate that owns the concept: `litellm-framing` for framing, `litellm_llms::Error` for the transformation layer

`litellm_llms::Error` (`crates/llms/src/error.rs`) is the one transformation error for every provider and API. `base_llm/ocr/error.rs` is the recorded exception until OCR folds into it

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or callback execution of any kind. Core runs each route as a machine that yields host operations and call events; which integrations consume those events is the host's business.

## Response caching and accounting boundary

Attach a `litellm_cache_response::ScopedCache` with `route.with_cache(cache)`. Cached and uncached routes use the same `execute` and `machine` methods. `CallOptions` carries a scope-free `CachePolicy` and observation; per-call policy never replaces the attached scope or service

Messages groups per-call dependencies in `CallContext` and explicitly sequences cache lookup, provider execution, result acceptance, and cache storage. Provider transport does not own cache orchestration. Stream capture remains in the shared cache implementation

Core owns request identity, typed response reconstruction and stream capture/replay. `cache-response` owns cache policy, namespacing, scope encoding, versioned envelopes and freshness. The SDK explicitly chooses shared scope. The gateway derives isolated scope from authenticated identity before attaching its service

Core delivers `ExecutionFacts` through the awaited `ResultReady` host operation for both provider and cached results, before public response processing or stream opening. Facts carry resolved model/provider and result source, including the hit key. Usage remains in the typed response or delivered stream, where completion and cancellation determine what was actually reported. Passive observation is not an accounting delivery mechanism

Core does not calculate prices, charge budgets, or update rate-limit counters. The legacy Python callback adapter translates execution facts into the existing Python logging contract; Python remains the accounting owner on that path. Native gateway accounting belongs to gateway dependencies, independently of `host-python`. Response-cache services expose no coordination counters or reservation APIs. A shared Redis deployment does not make response storage and accounting coordination the same dependency

Cache lookup follows provider preparation, credential resolution and the request interceptor. Keys describe the effective provider URL, authenticated headers and rewritten body. Signed requests bypass caching until the signing identity has a stable cache representation
