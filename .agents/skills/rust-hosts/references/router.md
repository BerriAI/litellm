# Router

## Python surface today

- `litellm.Router` in `litellm/router.py`. Before the refactor, anything reachable was public, with 165 members used by 120 files outside the router
- The new API keeps 37 endpoint methods (`acompletion`, `aresponses`, ...) and adds eight entry points with 36 methods behind them
  - `registry()`, `settings()`, `health()`, `select()`, `apply()` and `aapply()`, `observe()`, `register_plugin()`, `retire()`
- Reads go through `registry()` snapshots and every write goes through `apply(ChangeSet(...))`
- Old members forward through `DeprecatedRouterMembers` with a `DeprecationWarning` that names the replacement
- `tests/code_coverage_tests/router_surface_check.py` is a shrink-only allowlist of outside uses of router internals
- The router stops importing the proxy. Proxy-only questions go through a host the proxy installs

## Shared crates

- `crates/router`, `crates/router-core` and `crates/router-types`
- TBD: which entry point maps to which crate

## Per host

- python-sdk and python-proxy
  - `Router` keeps its public API and becomes a shim over the Rust router core
  - Routing plugins and strategies that exist only in Python: TBD
- rust-proxy
  - Builds the router from `config.yaml` and the database
  - A routing plugin Rust doesn't have makes the request forward to python-proxy

## Wire contracts

- Cooldowns, affinity and router usage in Redis follow [redis](redis.md)
- Others: TBD

## Order and gates

- TBD
