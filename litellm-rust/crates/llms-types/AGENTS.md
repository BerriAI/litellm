# Shared inference data contracts

`litellm-llms-types` owns shared inference API data contracts and their serialization. A type belongs here when it describes a request, response, event, header, or value that consumers must agree on independently of how a call executes. Being public, serializable, or used by several crates is not sufficient

These are intended ownership boundaries, not a claim that every existing item follows them. Apply them to new and changed contracts without unrelated cleanup

## Ownership and dependencies

Put API format contracts under `formats` and provider-specific wire contracts under `providers`. Follow the local [format rules](src/formats/AGENTS.md) and [provider rules](src/providers/AGENTS.md). Provider types may reuse format types. Format types must not depend on provider types. Keep format-independent data such as `headers`, `reasoning`, and `recognized` at the crate root

Shared inputs such as `ProviderSpecificHeader` and `ProviderSpecificHeaders` belong at the crate root. Selecting entries for a provider belongs in `core-utils`, and applying headers belongs in `http`

Keep one canonical definition and public import path per contract. When moving a type, update consumers together instead of leaving duplicate models or compatibility re-exports. A private submodule may expose its own items at its module root

Keep this crate at the bottom of the dependency graph. Use data, serialization, and optional schema libraries. Do not depend on execution crates, async runtimes, transport clients, or Python bindings. Python-compatible numeric coercion adapters belong in `python-compat` behind `serde-compat`, with field annotations here selecting where to apply them

No network or filesystem I/O, environment lookup, clock access, catalog lookup, or global configuration reads belong here. Defaults describe the data contract rather than runtime policy

## Data behavior

Allow deterministic constructors, accessors, serialization, schema generation, shape validation, and exact value conversions. Clamping effort, choosing thinking budgets, rewriting content, mapping finish reasons, computing normalized usage, and translating API formats belong in `llms` or the existing `core-utils` algorithms, even when they are pure functions

Use typed fields and enum variants for data consumers interpret or construct. Preserve opaque values where passthrough is part of the contract, including arbitrary tool arguments and JSON schemas. Keep one authoritative representation of each field, without duplicating a typed value in an extension map

Preserve each contract's treatment of unknown variants, extra fields, missing fields, explicit nulls, and numeric coercion. `Recognized<T>` retains values that fail typed parsing, including wrong-shaped values. Use it only where permissive passthrough is part of the contract. Typing an opaque field must neither reject previously accepted inputs nor accept malformed inputs previously rejected

## Execution boundaries

Call envelopes, prepared requests, live streams, and orchestration belong in `core`. Adapter traits, transformation contexts, capabilities, and transformer state belong in `llms`. Framing belongs in `framer`. Catalog records and pricing belong in `model-catalog`. Prompt normalization intermediates such as `Conversation` stay with their algorithms in `core-utils`

Host hooks, Python objects, credentials, clients, timeouts, routing decisions, and legacy logging operations do not become API payload types because they cross a crate boundary. Protocol error bodies may live here. Operational errors remain in the crate that raises them

## Verification

Follow the workspace test-placement and `rstest` rules. Test observable serialization, malformed-input rejection, unknown-data preservation, presence semantics, and value equality where relevant. Keep transformation, header-policy, and stream-execution tests in their owning crates. Never test import locations or source structure as substitutes for behavior

For Rust changes, run `cargo test -p litellm-llms-types` from `litellm-rust`. If changing schema-enabled types, also run `cargo test -p litellm-llms-types --features schema`. When moving contracts, check affected consumer crates as well
