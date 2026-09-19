litellm-core is the LiteLLM SDK in Rust — it makes the LLM call. Each top-level call is a module under `src/<route>/` exposing a public entrypoint named after the route (`messages::messages()`, the Rust equivalent of `litellm.messages()`): you call it and get a typed non-streaming response back.

## Crate layering

Each crate mirrors one top-level Python package, so a Rust path reads as its Python path with the crate name in place of the package directory. Dependencies only point down:

- `litellm-types` mirrors `litellm/types/`: pure serde data, no I/O
- `litellm-core-utils` mirrors `litellm/litellm_core_utils/`: pure helpers (provider resolution, prompt factory, call arguments), no network I/O
- `litellm-llms` mirrors `litellm/llms/`: `base_llm/<api>/transformation.rs`, `<provider>/<api>/transformation.rs`, and `custom_httpx/` (HTTP helpers and the OCR request handler)
- `litellm-core` mirrors the route packages (`litellm/ocr/`, `litellm/messages/`, ...): entrypoints, route request types, provider dispatch, the route machine, and hooks

A route module owns the call entrypoint, route request types (`*Request<'a>`), credential fallback, provider dispatch, and the handler glue that runs a provider config. Provider code never imports from core; when it needs the caller's hooks mid-call it goes through `litellm_llms::custom_httpx::llm_http_handler::CallHooks`, which each route implements over its host. Import every item from its canonical path. Never re-export another crate's items or give an item a second public path; the only re-export allowed is a private submodule surfacing its item at its module root (`mod error; pub use error::Error;`). Handlers belong in core or llms, never in a host crate

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or callback execution of any kind. Core runs each route as a machine that yields host operations and call events; which integrations consume those events is the host's business.
