litellm-core is the LiteLLM SDK in Rust: it makes the LLM call. Each top-level call is a module under `src/<route>/` exposing a public entrypoint named after the route (`messages::messages()`, the Rust equivalent of `litellm.messages()`): you call it and get a typed non-streaming response back. New and migrated entrypoints accept reusable HTTP clients from their hosts

A route module owns the call behavior: types, the provider template trait, provider transforms (under `providers/`), provider/auth/URL resolution, and the handler that performs the HTTP call. Runtime resources such as HTTP clients are injected. Handlers belong here, never in a host crate

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or callback dispatch. Env reads are limited to credential fallback in a route's `prepare.rs`.

Routes (messages, ocr, realtime) and providers (anthropic, mistral, openai) are modules, not crates.
