# structure

- `src/base_llm/<format>/` owns adapter contracts and shared transformation machinery
- `src/<provider>/<format>/transformation.rs` owns provider policy and Python counterparts; names identify counterparts, not full parity
- `src/error.rs` owns transformation errors; follow workspace test placement and `rstest` rules

# boundaries

- Shared API data and serialization belong in `llms-types`; call orchestration belongs in `inference-<format>`
- Providers choose payload, authentication and header policy; `http`, auth crates and `framer` apply transport, credential and framing mechanics
- Shared contracts must not import provider implementations; provider policy stays with its owner even when other providers reuse it
- Generic settings lookup and shared prompt conversion belong in existing `core-utils` helpers

# invariants

- Preserve unknown values, explicit nulls, field precedence and passthrough during layout or typing changes
- Use trait defaults for unchanged behavior and explicit delegation for shared provider policy
- Keep pure payload rewrites in transformations; contexts contain only inputs the adapter needs
- Preserve auth precedence, headers, serialization, provider selection and fallback gates with observable tests

# gotchas

- A schema-looking object inside `const`, `enum` or `examples` is literal data; schema rewrites must not traverse it
- Python parity can change an existing Rust contract, including forwarded credentials; make that compatibility change explicit
- Folder names and public helpers alone do not prove runtime reachability; trace the actual caller

# skills

- For provider transformations, layout alignment and Python test ports, use [.agents/skills/rust-provider-transforms/SKILL.md](.agents/skills/rust-provider-transforms/SKILL.md)
- Keep folder guides concise with `# structure`, `# boundaries`, `# invariants`, `# gotchas`, `# known gaps` and `# references`, in that order; omit empty sections
- Read existing folder guides before changing them; migrate useful constraints instead of discarding them
- Document only local rules, unresolved gaps and reasons that code cannot communicate; keep fuller examples in skill references

# validation

- Run affected named cases first, then the relevant provider/format suite and required repository checks
- Keep typed wire behavior tests in `llms-types`, provider behavior here, and Python-only settings projection in `tests/unit/rust_bridge/<format>`
- Record actual verification and remaining gaps; do not treat missing integration or review results as passes
