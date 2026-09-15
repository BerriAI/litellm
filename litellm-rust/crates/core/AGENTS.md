litellm-core is the LiteLLM SDK in Rust — it makes the LLM call. Each top-level call is a module under `src/<route>/` exposing a public entrypoint named after the route (`messages::messages()`, the Rust equivalent of `litellm.messages()`): you call it and get a typed non-streaming response back.

A route module owns everything the call needs: types, the provider template trait, provider transforms (under `providers/`), provider/auth/URL resolution, and the handler that performs the HTTP call. Handlers belong here, never in a host crate.

Not here: serving HTTP (axum routes, extractors), config file reading, rollout state, databases, or host-specific callback execution. Core owns lifecycle sequencing and callback payload construction; hosts execute the selected integrations. Env reads are limited to credential fallback in a route's `prepare.rs`.

Routes (messages, ocr, realtime) and providers (anthropic, mistral, openai) are modules, not crates.

Provider ports mirror the Python path under `litellm/llms/`, but they use Rust structure rather than copying Python inheritance. Keep provider transformation files flat by default: imports, constants and wire types, concrete configs and trait implementations, private helpers, then one test module. Add a production submodule only when it creates a real privacy, conditional compilation, or reuse boundary. Share behavior with private functions or explicit delegation; add a provider-specific base trait only when callers need that interface.
