# Adding a provider / route to litellm-rust

Three layers, same for every route (see `ocr` and `realtime` as references):

1. **Transform contract (pure)** — `crates/core/src/<route>/transformation.rs`: a `…ProviderConfig` trait (URL build + request/response transforms) + types in `types.rs`. No network, env, or auth.
2. **Provider config (pure)** — `crates/providers/src/<provider>/<route>/transformation.rs`: implement that trait as a `const <PROVIDER>_<ROUTE>_CONFIG`, mirroring the Python provider tree. Add parity unit tests.
3. **HTTP / transport (the host)** — `crates/providers/src/<route>.rs` (e.g. `ocr.rs`, `realtime.rs`): the callable fn (`run_ocr`, `realtime`). It resolves the key, builds the auth header, builds URL + transforms via the config, then does the network call. This is the only layer allowed to do I/O.

## Coding standards

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

**Calling:** the host invokes the route fn — the Python bridge calls `run_ocr`; the `ai-gateway` server calls `realtime`. Register new modules in `lib.rs` / `mod.rs`, then run `cargo fmt && cargo clippy --workspace -- -D warnings && cargo test --workspace`.
