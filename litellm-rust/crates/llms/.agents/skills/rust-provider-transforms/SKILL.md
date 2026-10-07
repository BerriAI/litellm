---
name: rust-provider-transforms
description: Implement or review Rust provider transformations and port their Python unit tests in litellm-llms. Use for adapter policy, wire shaping, auth/header parity or transformation layout; shared payload schemas belong to rust-wire-contracts.
---

# scope

Keep provider behavior in its owning adapter and preserve the requested API and execution boundaries

# workflow

1. Read the crate and affected folder guides, then pair the Python implementation and source assertions with the actual Rust caller
2. Read [transformation examples](references/transformation-examples.md) for layout, schema traversal, wire shaping, credentials or streaming
3. For test ports and parity claims, read [test-port examples](references/test-port-examples.md)
4. Make the narrow change needed by the demonstrated contract; verify observable inputs and outputs with named cases

# gotchas

- Implementing an `Unsupported` adapter can be legitimate port work; a failing test does not authorize unrelated features or overriding the user's report-first boundary
- Walk schema positions, not arbitrary JSON; preserve literal and opaque data
- Python parity and Rust compatibility are separate claims; trace boundary validation and auth precedence before changing either

# references

- [Crate rules](../../../AGENTS.md)
- [Transformation examples](references/transformation-examples.md)
- [Test-port examples](references/test-port-examples.md)
