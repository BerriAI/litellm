# ai-gateway

The Axum server fronting the Rust gateway. Owns transport and config; gateway authentication lives in `litellm-gateway-auth`, deployment selection is `core::router`, and the LLM call (transforms, provider auth headers, provider HTTP) is a `core` route entrypoint. No provider handler lives here.

- Routes follow one template: each module exposes `pub fn router() -> Router<AppState>`; `routes/mod.rs` only merges them. Simple routes are one file, non-trivial routes a folder (`handler`/`service`/`transport`). See `routes/AGENTS.md`
- Auth is an extractor from `litellm-gateway-auth`: add `RequireMasterKey` to handler args; never re-implement the check per route
- Handlers are thin: validate and delegate to `service`; no business logic, no provider calls, no transforms
- Services call `core`, they never reimplement it: pick the deployment, call the entrypoint. Provider resolution, auth headers, URL, and the HTTP call are `core`'s job
- State is shared and cheap to clone: long-lived handles behind `Arc` in `state.rs`; read env/config only in `main.rs`

## Auth (interim)

- Single master key (`LITELLM_MASTER_KEY`) enforced by `litellm_gateway_auth::RequireMasterKey`; `Authorization: Bearer <key>`
- Fails closed (500) when unset; constant-time compare; binds `127.0.0.1` by default (`HOST` to override)
- Per-key auth, budgets, rate limits delegated to the Python proxy later; health routes unauthenticated

## Python interop

- Python-backed loading lives in `litellm-config`, load-time only; `python-config` feature forwards to it
- The realtime data path never takes the GIL
