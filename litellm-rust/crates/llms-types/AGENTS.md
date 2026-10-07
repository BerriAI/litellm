# structure

- `formats/chat_completions`, `formats/responses` and `formats/ocr.rs` own shared API payloads
- Split larger formats by content, request, response and streaming; keep small contracts in one file
- `headers`, `json_schema`, `recognized` and `serde_compat` hold format-independent data helpers

# boundaries

- A type belongs here when it describes data consumers agree on independently of call execution
- Adapter contracts and policy belong in `llms`; call envelopes and execution state belong in `inference-<format>`
- Keep this crate free of execution dependencies, I/O, async runtime, environment, clock, catalog and global settings access
- Provider types may use format types; format types must not depend on provider types

# invariants

- Keep one canonical definition and import path; update consumers together rather than adding duplicate models or compatibility re-exports
- Preserve each contract's missing/null distinction, unknown fields, discriminator handling and malformed-input acceptance
- Keep one representation of a field; typed fields must not also live in an extension map
- Allow shape validation, constructors, accessors, schema generation and exact value conversions; defaults describe data, not runtime policy

# gotchas

- `Recognized<T>` preserves wrong-shaped values as well as unknown variants; use it only where permissive passthrough is intended
- A format field does not promise provider support; capability checks, clamping, normalization and cross-format conversion belong in adapters
- JSON Schema `const`, `enum` and `examples` contain literal data, not child schemas
- Normalized host responses can be partial LiteLLM contracts; replacing them with upstream schemas can change behavior

# skills

- For wire types, ownership decisions and Serde changes, use [.agents/skills/rust-wire-contracts/SKILL.md](.agents/skills/rust-wire-contracts/SKILL.md)
- Keep folder guides concise with `# structure`, `# boundaries`, `# invariants`, `# gotchas`, `# known gaps` and `# references`, in that order; omit empty sections
- Read existing folder guides before changing them; migrate useful constraints instead of discarding them
- Keep detailed ownership examples and serialization cases in skill references

# validation

- Test round trips, missing/null distinctions, unknown-value preservation and malformed-input rejection
- Follow workspace test placement and named `rstest` cases; test behavior rather than source layout or import paths
- Run `cargo test --manifest-path litellm-rust/Cargo.toml --locked -p litellm-llms-types --features schema` from the repository root
