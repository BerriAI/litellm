---
name: redis-access
description: Add Redis-backed operations, hide existing redis-py access behind typed interfaces, or explicitly port implementations to Rust without expanding migration scope
---

# Redis access

## Adding new Redis-backed access

- Callers depend on typed operations, never redis-py
- Reuse an existing cache or state interface when it fits
- Otherwise add a small operation interface beside the code that owns the state
  - Express the caller's need, such as reserving capacity or draining pending updates
  - Use a narrow Python `Protocol` when an injected dependency needs a contract
  - Do not mirror the whole redis-py API or grow a universal Redis wrapper
- Give callers typed inputs, results, and backend-neutral failures
  - Do not expose clients, pools, pipelines, script handles, Redis reply objects, or redis-py exceptions
  - No `get_client()`, generic `execute_command()`, or callback receiving a raw client
  - A protocol returning a redis-py pipeline still leaks the implementation
- Keep redis-py imports, command execution, reply decoding, and error translation inside the implementation
- Keep key layout, value formats, TTL policy, and fallback decisions with the owning cache or state component
  - The transport applies the namespace once and executes the supplied TTL
- redis-py is fine as the implementation, adding an operation does not require a Rust port

## Asked to move existing access behind an interface

- Find the calls being moved, including raw client attributes, client factories, script registrations, and exception handling
- Identify the complete operation and its atomicity boundary before extracting it
- Move the existing redis-py implementation behind the relevant operation
  - Keep scripts and transactional command sequences together
  - Move script registration and invocation together, not just client construction
- Replace direct client access with the operation and inject the dependency
- Preserve observable behavior with focused regression tests
  - Keys, namespaces, value encoding, TTL units and refresh behavior
  - Ordering, missing values, partial failures, and atomicity
  - Existing fail-open or fail-closed policy and caller-visible errors
- Translate errors at the implementation boundary without hiding an outage as a cache miss
- Keep this an extraction, not a backend migration or unrelated cleanup
- Leave unrelated legacy access unchanged, but add no new raw-client escape hatches

## An equivalent Rust operation already exists

- Leave it and the current backend selection unchanged unless integration is explicitly requested
- Its existence alone does not justify switching a Python caller to Rust
- Interface extraction can keep using redis-py even when the Rust operation is available

## Asked to port an implementation to Rust

- Reuse `crates/redis-client` for transport and follow its `AGENTS.md`
  - Keep domain operations, keys, value formats, TTL policy, and fallback outside the transport crate
  - Keep redis-rs types and raw replies out of domain-facing and Python-facing contracts
- Resolve configuration in `crates/config` and compose clients at the gateway boundary
  - Preserve coordination, transaction-buffer, and response-cache role selection
  - Gateway client construction alone does not wire request consumers
- Keep Python conversion and native-handle construction at the Python bridge boundary
- Switch callers through the same operation interface only when integration is requested
- Verify equivalent behavior against real Redis before switching callers
  - Check supported standalone and cluster topologies, slot placement, and namespace compatibility
  - Check TTLs, encoding, missing values, concurrency, errors, and outage behavior
  - Rust pipelines are non-atomic, never substitute one for a transaction or atomic script
- Never retry an ambiguous write through another backend, a timeout may follow a committed increment, pop, or lock
- Remove the old implementation only when no callers need it

## Migration direction

- Centralize access behind typed operations first, then replace implementations incrementally
- Callers should not change when redis-py is replaced by Rust
- Inject interface fakes for caller tests and test implementations against real Redis
- Keep client lifecycle, connection tuning, and topology handling inside infrastructure implementations
- Mixed Python and Rust deployments must agree on shared keys, namespaces, value formats, and atomic operations
- Redis client configuration, caller migration, and redis-py removal are separate steps
