## Layering

- `gateway-inference` (serving) -> `inference-<fmt>` (one API's call) -> `inference` (shared base)
- This crate is the shared base every format crate builds on
  - `RouteError` (`src/error.rs`), `CallOptions` (`src/lib.rs`), `CallContext` (`src/context.rs`)
  - diagnostic spans (`src/diagnostic.rs`), outbound send and signing (`src/outbound.rs`), provider resolution (`src/provider.rs`), `CoreResources` (`src/resources.rs`)
  - response caching (`src/caching.rs`): `Cachable`, `StreamCachable`, `CacheRequest`, `CallCache`, `execute_unary`, `execute_streaming`, stream capture
  - Shared test helpers belong in litellm-inference-testing, used only as a dev dependency
- Nothing here names an API format; format-specific code, constants, tests and test builders live in their `inference-<fmt>` crate
- Never depend on an `inference-*` crate from here

## Format crates

- `inference-chat`, `inference-messages`, `inference-responses`, `inference-ocr`, `inference-transcription`, one per API
- Each owns its entrypoint, route request types (`*Request<'a>`), credential fallback, provider dispatch, and the handler glue that runs a provider config
- Each owns its `Protocol`, its `Cachable` / `StreamCachable` impl, its constants, and its tests
- `inference-*` is orchestration only
  - shared API data contracts belong in `litellm-llms-types`
  - adapter contracts and shared transformation machinery in `llms/src/base_llm/<format>/`
  - provider policy in `llms/src/<provider>/<format>/`
- Select concrete adapters, then invoke their contracts instead of applying one provider's policy to every call
- Route types describe call envelopes and execution state, not duplicate public payload schemas
- Format-specific rules go in that crate's `AGENTS.md`

## Routes

- Hosts assemble route objects from shared `CoreResources`, HTTP settings, and secret sources
  - each route owns its provider client and authentication dependencies
  - gateway routes live for the gateway lifetime; Python assembles routes per call from its settings snapshot
- Calls pass `Interceptors` and an optional `ObservationSender` separately
  - use `&()` for no hooks and `None` for no observer
  - handlers accept `Interceptors`, never a concrete `ChannelInterceptors`
- Construction does no work; preparation and lifecycle observation begin when the future is polled
- Native observers receive start and terminal events through the shared call runner; a stream retains its lifecycle until exhaustion, error, or drop. Hosted routes leave terminal observation to their driver
- A hosted format's `route.rs` declares the concrete `Protocol` and a route method that takes a typed request and builds a `litellm_host::call::HostedMachine` with `hosted_call`; audio transcription has no hosted machine and returns its response directly
  - the shared call plumbing owns stream opening, delivery, backpressure, and detachment
  - request decoding belongs to the boundary before the machine starts
  - route closures only supply execution dependencies and route-specific host capabilities
  - Python uses its own shared driver and preserves caller-task callback execution
- Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or callback execution of any kind. Routes run as machines that yield host operations and call events; which integrations consume those events is the host's business

## Crate boundaries

- Dependencies only point down; Python package names identify counterparts, not ownership
- `litellm-llms-types` owns shared inference API contracts, grouped by format: pure serde data and shape validation, no I/O
- `litellm-core-utils` mirrors `litellm/litellm_core_utils/`: pure helpers (provider resolution, prompt factory, call arguments, settings lookup and layer merge), no network I/O
- `litellm-http` is Rust-only and route-neutral: settings resolution, pooled `reqwest` clients, TLS, proxies, the SSRF-safe media fetcher, request and header helpers, and transport errors
- `litellm-llms` mirrors `litellm/llms/`: `base_llm/<api>/transformation.rs`, `<provider>/<api>/transformation.rs`, and `base_llm/ocr/handler.rs`
- `litellm-inference-<fmt>` mirrors the route packages (`litellm/ocr/`, `litellm/messages/`, ...)
- Provider code never imports from `inference` or `inference-*`
- Handlers belong in `inference-*` or `llms`, never in a host crate
- Import every item from its canonical path. Never re-export another crate's items or give an item a second public path; the only allowed re-exports are a private submodule surfacing its item at its module root (`mod error; pub use error::Error;`) and each format crate's `pub use litellm_inference::RouteError as Error;`, which keeps the pre-split `<fmt>::Error` name

## Error placement

- The workspace `Error definitions` rules shape each crate's error; this section decides which crate and module a failure belongs to
- A failure is declared once, by the lowest crate that raises it
  - every crate above nests it unchanged (`#[error(transparent)] Auth(#[from] litellm_auth::Error)`) or maps it once at its boundary, as `src/error.rs` does for `litellm_llms::Error`
  - `RouteError` collects route failures for every format crate and never re-declares a variant a lower crate raises
- Scope follows the concept, not the first caller
  - an error type under `litellm-llms`'s `<provider>/` is private to that provider: no other provider and nothing in `base_llm` may import it
  - a failure two providers or two routes can hit (wire framing, stream event decoding, a malformed provider response) belongs to the crate that owns the concept: `litellm-framer` for framing, `litellm_llms::Error` for the transformation layer
- `litellm_llms::Error` (`crates/llms/src/error.rs`) is the one transformation error for every provider and API; `base_llm/ocr/error.rs` is the recorded exception until OCR folds into it

## Response caching and accounting

- Attach a `litellm_cache_response::ScopedCache` with `route.with_cache(cache)`; cached and uncached routes use the same `execute` and `machine` methods
- `CallOptions` carries a scope-free `CachePolicy` and observation; per-call policy never replaces the attached scope or service
- Provider transport does not own cache orchestration; stream capture stays in `src/caching.rs`
- This crate and the format crates own request identity, typed response reconstruction, and stream capture and replay
- `cache-response` owns cache policy, namespacing, scope encoding, versioned envelopes, and freshness
- The SDK explicitly chooses shared scope; the gateway derives isolated scope from authenticated identity before attaching its service
- `ExecutionFacts` are delivered through the awaited `ResultReady` host operation for both provider and cached results, before public response processing or stream opening
  - facts carry resolved model/provider and result source, including the hit key
  - usage remains in the typed response or delivered stream, where completion and cancellation determine what was actually reported
  - passive observation is not an accounting delivery mechanism
- Inference does not calculate prices, charge budgets, or update rate-limit counters
  - the legacy Python callback adapter translates execution facts into the existing Python logging contract; Python remains the accounting owner on that path
  - native gateway accounting belongs to gateway dependencies, independently of `host-python`
  - response-cache services expose no coordination counters or reservation APIs; a shared Redis deployment does not make response storage and accounting coordination the same dependency
- Cache lookup follows provider preparation, credential resolution, and the request interceptor
  - keys describe the effective provider URL, authenticated headers, and rewritten body
  - signed requests bypass caching until the signing identity has a stable cache representation
