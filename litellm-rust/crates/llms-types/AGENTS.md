# Shared inference data ownership

`litellm-llms-types` owns shared inference API data contracts and their serialization. A type belongs here when consumers must agree on its request, response, event, header, or value semantics independently of how a call executes. Being public, serializable, or used by several crates is not sufficient

Put API format contracts under [`formats`](src/formats/AGENTS.md), provider-specific wire contracts under [`providers`](src/providers/AGENTS.md), and format-independent data at the crate root. Provider types may reuse format types. Format types must not depend on provider types

Keep one canonical definition and public import path per contract. Update consumers together when moving a type, without duplicate models or compatibility re-exports. A private submodule may expose its own items at its module root

## Data versus execution

This crate may own constructors, accessors, serialization, schema generation, shape validation, and exact value conversions. Defaults describe the data contract rather than runtime policy

Provider capabilities, parameter mapping, payload rewriting, usage normalization, and cross-format translation belong in `llms` or the existing `core-utils` algorithms. Call envelopes, prepared requests, and orchestration belong in `core`. Live streams and transformer state belong with their execution machinery, while stream-event payloads belong here

Catalog records and pricing belong in `model-catalog`. Prompt normalization intermediates belong with their algorithms in `core-utils`. Host hooks, credentials, clients, timeouts, Python objects, and logging operations belong in their owning execution crates. Protocol error bodies are data contracts, while operational errors belong to the crate that raises them

Shared input data such as `ProviderSpecificHeaders` belongs here. Selecting entries for a provider belongs in `core-utils`, and applying headers belongs in `http`. Python-compatible numeric coercion belongs in `python-compat` behind `serde-compat`, with field annotations here selecting where it applies

Keep dependencies limited to data, serialization, optional schema libraries, and data-only compatibility helpers. No execution-crate dependencies, async runtimes, transport clients, Python bindings, I/O, environment reads, clock access, catalog lookups, or global configuration reads

## Contract preservation

Use typed fields and enum variants for data consumers interpret or construct. Preserve opaque values where passthrough is part of the contract, including arbitrary tool arguments and JSON schemas. Keep one authoritative representation of each field, without duplicating typed values in extension maps

Preserve existing unknown-value, extra-field, missing-field, explicit-null, and numeric-coercion behavior. `Recognized<T>` belongs only where permissive passthrough is already part of the contract. Adding types must neither reject previously accepted inputs nor accept malformed inputs previously rejected

Serialization and value-semantics tests belong here. Transformation, header-policy, and execution tests belong in their owning crates. Follow the workspace test rules

These boundaries apply to new and changed contracts. Existing misplaced types do not justify new ones or require unrelated cleanup
