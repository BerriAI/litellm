---
name: rust-wire-contracts
description: Define or revise shared Rust Chat Completions, Responses and OCR payloads and their Serde contracts in litellm-llms-types. Use for canonical type ownership, missing/null semantics, discriminators and unknown-field preservation; adapter policy and provider transformations belong in llms.
---

# scope

Represent shared wire data independently of how calls execute

# workflow

1. Read the crate and affected format guides and identify the existing public contract and its consumers
2. Read [ownership and wire examples](references/ownership-and-wire-examples.md) when placing types or changing serialization
3. Preserve acceptance, omission, explicit nulls and unknown data; keep one canonical type and update consumers together
4. Verify round trips and rejection behavior in this crate; verify provider policy in its adapter

# gotchas

- `Recognized<T>` is permissive even for malformed values; adding it can loosen validation
- Normalized LiteLLM responses need not be complete upstream schemas
- Schema literals and opaque extension values remain data; provider normalization stays outside this crate

# references

- [Crate rules](../../../AGENTS.md)
- [Ownership and wire examples](references/ownership-and-wire-examples.md)
