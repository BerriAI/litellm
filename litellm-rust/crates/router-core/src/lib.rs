//! Routing for LiteLLM: decide which deployment serves a request, and drive retries, fallbacks
//! and cooldowns around the attempts.
//!
//! This crate defines the boundary `litellm.Router` is moving behind. Today the Python router in
//! `litellm/router.py` implements all of it; this crate is where the Rust core will implement the
//! same interface. Until then it holds this overview and nothing else, so the interface is written
//! down in code before either side changes.
//!
//! # The public surface
//! Code outside the router uses `litellm.Router` through two kinds of members and nothing else:
//! - The endpoint methods (`acompletion`, `aembedding`, `aresponses`, `aanthropic_messages`, the
//!   files, batches, assistants and vector store wrappers, 37 in all). Each is a thin wrapper over
//!   `call(op, request)`.
//! - Eight entry points: `registry()`, `settings()`, `health()`, `select()`, `apply()`,
//!   `observe()`, `register_plugin()` and `retire()`.
//!
//! Every other member that outside code uses today has a named replacement among those. The old
//! names stay available for about a month through a `DeprecatedRouterMembers` mixin, which
//! implements them over the entry points and warns on use. A CI check
//! (`tests/code_coverage_tests/router_surface_check.py`) keeps an allowlist of the members outside
//! code touches; the list may only shrink.
//!
//! # Structure
//! - `Registry`: the configured deployments, indexed by id, model group, alias, wildcard pattern
//!   and team. Immutable and versioned. Seven queries: `deployment(id)`, `deployments(filter)`,
//!   `names(filter)`, `resolve(name, scope)`, `group(name)`, `access_groups()` and
//!   `credentials(id)`. "Registry" is the router's view of what is configured; per-model pricing
//!   and capabilities are the model catalog (`litellm-model-catalog`), which the router reads
//!   through its dynamic config.
//! - `RouterConfig` / `RouterDynamicConfig`: everything the router needs to know, passed in. The
//!   router never reads process globals. Static config (model list, strategy, retry, timeout and
//!   cooldown policy, Redis settings, plugins) is fixed for the router's life. Dynamic config
//!   (pricing, model aliases, fallbacks, retry counts, callbacks) is a versioned snapshot read
//!   once per call.
//! - `Settings`: one typed, frozen view of the configuration a caller may read.
//! - `Health`: cooldowns, blocked deployments and usage, kept in shared cache keys so Python and
//!   Rust workers see the same state. `report(id, outcome)` records outcomes the router did not
//!   see itself (a health check, an interceptor that marks a deployment failed).
//! - `select(model, request)`: resolve a name, filter eligible deployments, pick one with the
//!   configured strategy, without calling it.
//! - `apply(changes)`: one `ChangeSet` (upserts, deletes, settings), applied atomically, producing
//!   a new registry version.
//! - `call(op, request)`: select, run the attempt, and retry or fall back until it succeeds or the
//!   policy is exhausted.
//!
//! # Execution
//! Each attempt resolves an `Executor` after a deployment is picked. A Rust-native operation runs
//! its `inference-*` route machine in-process; anything else goes through `Host::invoke`. A
//! started attempt is never replayed on a different executor.
//!
//! # Hosts
//! `Host` is what the embedding process provides: `invoke` for non-native operations, `select`
//! for strategies and sub-routers not yet ported, `allow_fallback` for access and budget checks,
//! `sleep`, `observe` for routing events, and `cache`, a DualCache-style store that holds all
//! routing state under the same keys, encodings and TTLs the Python router uses. `PythonHost`
//! serves them from the existing Python code when the router runs inside Python; `NativeHost`
//! serves them in the Rust server, where an operation without a Rust implementation makes the
//! config fail at build time, never partway through a request.
//!
//! # Plugins
//! Routing strategies, pre-call checks and sub-routers (auto, complexity, quality, adaptive)
//! register through `register_plugin` and see a `RoutingContext` (registry, settings, health and
//! the current request), never the router itself.
