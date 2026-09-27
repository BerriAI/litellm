litellm-core is the LiteLLM SDK in Rust. Each top-level call is a module under `src/<route>/` exposing a public entrypoint named after the route. `messages::messages()` returns `MessagesResponse::Message` for a completed response or `MessagesResponse::Stream { headers, chunks }` when the request sets `stream: true`. The chunks are Anthropic SSE bytes in a `Stream<Item = Result<Bytes, Error>>`. Dropping the stream cancels the call. The Python bridge drives `messages::route::messages_machine()` instead, because Python has to answer the call's operations on its own thread; the gateway and the Rust SDK call the plain entrypoint

A route module has the same five pieces, in the order Python runs them. `types.rs` holds the call, the provider request, and the response. `prepare.rs` resolves the provider and credentials and shapes the request (Python's `validate_environment`, `get_complete_url`, `transform_request`). `handler.rs` resolves auth, offers the wire request to `litellm_host::hooks::RouteHooks::before_send`, sends it, reports the raw response through `emit`, and normalizes the response or stream (`pre_call`, `post`, `post_call`, `transform_response`). `mod.rs` exposes the entrypoint that runs prepare then handler with no hooks (`()`). `route.rs`, where a host needs it, wraps the same two calls in a `CallMachine` whose `HostChannel` is the hooks, and pumps a stream through `open` and `deliver`. A handler takes `&impl RouteHooks<Error>` and never a `HostChannel` directly, so it runs without a coroutine. Keep provider transport and transformation details out of the machine driver

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
