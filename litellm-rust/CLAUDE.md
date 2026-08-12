# CLAUDE.md

This file defines the rules for Rust work in LiteLLM.

## Provider Coding Standards

Before writing new logic, look for an existing base to extend. When a change is
“the same behavior for one more provider/endpoint/integration”, the codebase
almost always already has a shared abstraction for it (for example, provider
`BaseConfig` transformation classes in `litellm/llms/base_llm/`, shared
helpers in `litellm_core_utils/`, typed request/response models, or factory
functions). Find it first with a search, then add the new variant by inheriting
from or composing that base, overriding only what genuinely differs (model
name, parameter mapping, or auth).

Never copy an existing implementation and edit it in place, and never hand-roll
a parallel version of logic a base already provides. If you catch yourself
writing a second copy of a pattern that exists twice already, stop and extract a
base instead: put the shared shape in one place and make both call sites thin
variants of it. The test for a good abstraction is that adding the next provider
is a few declarative lines, not a new file of duplicated flow. Only diverge from
the base when behavior is genuinely different, and say so explicitly in the PR.

## Crates (exactly three — see AGENTS.md)

`litellm-core` **is** the LiteLLM SDK in Rust: it makes the LLM call.
`litellm-ai-gateway` is an HTTP/WebSocket server in front of it, and
`litellm-python-bridge` exposes it to the Python SDK. A crate is a **layer**, not
a route — add modules, not crates.

## Core Boundary

`litellm-core` owns the whole call. The Rust equivalent of `litellm.messages()`
is `litellm_core::messages::messages(request).await`: you call it, it does the
provider call, and you get a typed non-streaming response back.

Route-level Rust structure mirrors LiteLLM's Python responsibilities:
- `core/src/<route>/` owns the route end to end: the public entrypoint fn named
  after the route in `mod.rs`, the request/response types (`types.rs`), the
  provider template trait (`transformation.rs`), the provider/auth/URL
  resolution (`prepare.rs`), the HTTP client (`client.rs`), and the handler that
  performs the call (`handler.rs`). `core/src/messages` is the reference.
- `core/src/providers/<provider>/<route>/transformation.rs` owns the
  provider-specific transform. For Anthropic Messages, this means
  `core/src/providers/anthropic/messages/transformation.rs`.
- Handlers live in `core`, never in a host. `ai-gateway` must not contain a
  route handler that talks to a provider; its axum route reads the HTTP request,
  picks a deployment, and calls the `core` entrypoint. `python-bridge` marshals
  Python objects and calls the same entrypoint.

Streaming keeps the same shape: the route entrypoint has a `<route>_stream`
variant in `core` that returns the upstream response so a host can splice it to
its own caller; the host still owns no provider logic.

Call-hook and lifecycle instrumentation, including phase timing, usage
accumulation, and callback payload construction, always lives in `core`.
Hosts feed observed events into core and dispatch the completed payloads through
their I/O logger; hosts must not own callback orchestration.

Allowed in `core`:
- The public entrypoint for a top-level LiteLLM call
- Request/response transforms and stream chunk normalization
- Provider resolution, auth header construction, and URL building
- The provider HTTP call itself, through a shared reused client with connect and
  request timeouts
- Shared data types and validation errors
- Deterministic token/cost helper logic

Not allowed in `core`:
- Serving HTTP: axum routes, extractors, and transport concerns stay in the host
- Filesystem access
- Database access
- Config file reading and rollout state
- Logging callbacks, spend writes, or custom callbacks
- Global mutable runtime state

Env reads in `core` are limited to credential fallback inside a route's
`prepare.rs` (the `env_lookup` closure), mirroring what the Python SDK does when
no key is passed. Everything else config-shaped is resolved by the host and
passed in.

Routes still hosted in `ai-gateway` (`ocr`, `audio_transcription`, `realtime`)
predate this rule and are being moved into `core` route modules; do not add new
ones there, and prefer moving one when you touch it.

Python owns rollout state and fallback while Rust is being introduced. Rust
paths must be off by default until parity tests prove equivalence with Python.
A new provider/route may instead be implemented rust-only with no Python
reference; then the Python interface is a thin dispatch that calls Rust with no
fallback, and you state the rust-only choice explicitly in the PR. Either way
the Python side stays minimal (it only marshals inputs and calls the Rust
interface), never add a per-route feature flag, and never push provider
dispatch into `litellm/main.py`; put it in a thin dispatch class under
`litellm/llms/<provider>/<route>/`.

## Production Bar

Rust code in this workspace is held to a strict parity and robustness bar from
the first PR:

- Correctness parity is proven with tests. Do not rely on README claims or
  manual inspection for a port that mirrors Python behavior.
- Every provider transform must have unit tests for supported-parameter
  filtering, request body shape, response normalization, missing/null fields,
  and bad-input errors.
- When Rust is exposed through Python, add Python tests that prove disabled,
  enabled, and unavailable-bridge fallback behavior.
- Avoid panics on user/provider input. Return typed errors and let the host map
  them to Python exceptions or HTTP responses.
- OCR handles documents that often contain personal data. Do not log document
  contents, base64 payloads, provider response bodies, or secrets.
- Error messages must be useful but data-minimized. Truncate or sanitize any
  upstream body before it crosses a host boundary.
- Treat empty or whitespace-only credentials, URLs, and config values as absent
  at the host/config resolution layer.
- Preserve Python output shape intentionally. If a field is always serialized as
  `null` for Python parity, leave a short comment explaining that parity choice.

## Network I/O Rules

These rules apply to every module that executes network I/O, whether it is a
`core` route handler or a host such as `ai-gateway`:

- Set connect and full-request timeouts. No unbounded waits.
- Reuse HTTP clients; do not construct clients per request.
- Prefer rustls TLS for portable Python wheels and Linux images unless there is
  a documented reason not to.
- Add request IDs and structured tracing at the host layer, without logging OCR
  document contents or secrets.
- Do not echo raw upstream response bodies to callers. Sanitize and bound them.
- Avoid `expect`/`unwrap` in server startup and request paths unless the panic is
  impossible by construction and documented.

## Rust Style Guide

All Rust in `litellm-rust/` follows the official Rust Style Guide:
https://doc.rust-lang.org/style-guide/

`rustfmt` implements the guide's formatting rules by default, so the mechanical
side is enforced for you: run `cargo fmt` before committing and CI gates every
PR on `cargo fmt --check` (see Checks). Do not hand-format against rustfmt or add
a `rustfmt.toml` that diverges from the default style; the default style *is* the
guide.

The guide also covers conventions rustfmt cannot auto-apply; follow these too:
- Naming: `snake_case` for items, functions, and modules; `UpperCamelCase` for
  types, traits, and enum variants; `SCREAMING_SNAKE_CASE` for constants and
  statics; acronyms count as one word (`HttpClient`, not `HTTPClient`).
- Ordering and grouping the guide prescribes: imports grouped std / external /
  crate-local, derives before other attributes, and consistent item order.
- Idioms the guide recommends over the formatter fighting you (e.g. prefer
  restructuring an over-long expression rather than forcing an awkward wrap).

## Constants

Magic numbers and fixed strings go in a crate-level `constants.rs`, never
hardcoded inline — the Rust mirror of Python's `litellm/constants.py`.

- Each crate that needs them has `src/constants.rs` (declared `mod constants;`);
  import from it (`use crate::constants::...`). Don't scatter `const` values at
  the top of feature modules.
- An env-overridable tunable still lives in `constants.rs` as its `DEFAULT_*`
  value; the env read (with fallback to that default) happens at the host/config
  resolution layer, not in `core`/`providers`.
- Exception: a value that is purely local to one function and has no meaning
  elsewhere may stay inline, but prefer `constants.rs` when in doubt.

## Checks

Run these before pushing Rust changes. The same checks run in GitHub Actions
for changes under `litellm-rust/`.

```bash
cd litellm-rust
cargo fmt --check
# the ai-gateway binary + server code is behind the `server` feature
cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings
cargo clippy -p litellm-core -p litellm-python-bridge --all-targets -- -D warnings
cargo test --workspace
```

When a Rust path is exposed through Python, add Python parity tests that compare
the existing Python output with the Rust-backed output.
