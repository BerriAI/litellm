---
name: rust-hosts
description: Decide where code goes between the shared Rust crates, python-bridge and Python for the python-sdk, python-proxy and rust-proxy hosts, design a public API Rust can implement, move a module into a shared crate, or prove parity between old and new code. Use when adding a pyclass or host call in python-bridge, porting a module, refactoring a module behind an entry point, or touching cache, Redis, database, secret manager, router, model catalog, HTTP client or callback code on the Rust path
metadata:
  internal: true
---

# Rust hosts

One implementation lives in shared Rust crates, and each host runs it. python-sdk on the native extension is a permanent host, so design for it as an end state, not a step on the way to rust-proxy

These are migration rules and target boundaries, not a claim that every subsystem is already ported. `crates/` paths are relative to `litellm-rust/`. A `TBD` is an unresolved decision, not permission to invent a new seam

## The three hosts

Read the page for each host the change runs on

- [python-sdk](references/python-sdk.md)
  - Permanent. Runs the core through `python-bridge`, and may ask Python through host calls
- [python-proxy](references/python-proxy.md)
  - Runs behind rust-proxy for a long time and serves every request Rust can't serve yet. Retired once rust-proxy covers everything
- [rust-proxy](references/rust-proxy.md)
  - No interpreter. Sits in front, serves what it supports end to end in Rust, forwards the rest to python-proxy

## Subsystems

Read the page for each subsystem the change touches

- [cache](references/cache.md)
- [redis](references/redis.md)
- [database](references/database.md)
- [secret manager](references/secret-manager.md)
- [router](references/router.md)
- [model catalog](references/model-catalog.md)
- [http](references/http.md)
- [callbacks](references/callbacks.md)

## Classes and values

What Python exposes decides how the core reaches it

| Python exposes | python-sdk and python-proxy | rust-proxy |
|---|---|---|
| A class users configure or swap (`litellm.cache`, the secret manager, `credential_provider`) | The core makes a host call, and the Python object answers | The shared crate, built from `config.yaml` |
| Values (kwargs, `litellm.*` globals, config fields) | Read through a host call or passed in at construction | Read from `config.yaml` |

- The built-in implementations of those Python classes become thin shims over the shared crates. A Python object answering a host call may call back into Rust, and that is fine while the logic exists once
- Resolving values (Redis settings, cloud tokens) is always a shared crate, never Python
- Data Rust can load itself is loaded by Rust on every host, never asked of Python. The model catalog is read and refetched natively on the extension too
- A user-supplied Python object with no Rust equivalent stays SDK-only, with a clear error where it is unsupported

## Host calls and injected traits

- A host call is for anything the Python hosts may answer with Python, such as the cache, the secret manager and `litellm.*` settings. The core yields it through `crates/host` and the host's driver answers (`host-python` or `host-native`)
- An injected trait is for things that are Rust on every host, such as the Redis client, token providers, clocks and sinks. Built once at construction, never yielded per call
- The core never names Python in a shared type. Whether Python answers is the host's business
- Speed of the Python answer doesn't matter. Requests that need speed run end to end on rust-proxy

## Where code goes

| Code | Where it lives |
|---|---|
| Used by rust-proxy | A shared crate |
| Used by python-sdk and rust-proxy | A shared crate, plus a pyclass in `python-bridge` |
| Used only by python-proxy | Stays in Python. It serves the long tail until rust-proxy supports it |
| Python-owned state the core asks about | A `litellm/rust_bridge/host/` module, called only by the extension |

- Product behavior lives in shared crates and never needs Python to be correct
- Hosts differ in how input arrives, never in how it is resolved
- Follow the subsystem's migration gates. In particular, [database](references/database.md) moves operations behind repositories and ports them incrementally even while python-proxy remains

## Moving a module into a shared crate

1. Define a public API. Map what the module imports and what imports it, bucket each outside use by what the caller wants, then make DB and cache access, mutability and ownership explicit
2. Route every caller through it. The new API implements the old one, old names forward with a `DeprecationWarning` that names the replacement, and a shrink-only allowlist in CI stops new coupling (`tests/code_coverage_tests/router_surface_check.py` is the model)
3. Build the Rust crate behind that API
4. Make the Python module a thin shim over Rust
5. Prove exact parity between old and new
6. Delete the Python implementation. No fallback copy stays behind

Rebase often while a refactor is in flight, and check what landed on main for changes to behavior already moved

## Surfaces Rust can implement

| Avoid | Prefer |
|---|---|
| Shared mutable dicts | Typed values in, values out |
| Live references handed out | One owner, snapshot reads, writes through one `apply` |
| Hidden cache and DB access | Named reads and writes |
| `litellm.*` globals read at call time | Config passed in at construction |
| Callbacks that can touch the whole request | Hooks that declare what they read and write |
| Callers parsing error text | Typed errors |

## Proving parity

- Fuzz old against new with [BerriAI/litefuzz](https://github.com/BerriAI/litefuzz). Its README covers suites, levels, oracles and coverage
- Check public values, errors and side effects, including the subsystem's shared-state contracts
- For Redis and database ports, also run the compatibility checks against real Redis or PostgreSQL described in their reference pages
