# Rust design handoffs

Keep one design per Markdown file in this directory, using a descriptive kebab-case name. Start with status, scope, and the source revision inspected. A draft records a proposal, not an implemented API or an approved behavior change

Use the following sections so another contributor can pick up the work without the original conversation

| Section | Contents |
| --- | --- |
| Problem and evidence | Concrete failure modes and links to current code |
| Decision and invariants | Recommended approach, guarantees, and explicit limits |
| Ownership and file layout | Canonical definitions, dependencies, and files to change |
| Proposed API | Representative callers, types, and generated behavior |
| Compatibility | Existing inputs, precedence, and intended behavior changes |
| Implementation and verification | Small migration steps and observable acceptance criteria |
| Open decisions | Unresolved choices and what depends on them |

Keep examples marked as proposed until implemented. Use repository-relative links inside these documents and versioned links for library behavior. Update the same document as decisions settle. Split supporting material into a same-named directory only when the design needs actual fixtures or experiments

| Design | Status | Scope |
| --- | --- | --- |
| [Typed provider endpoints](typed-provider-endpoints.md) | Draft | Endpoint declarations, URL typestate, macros, and Python compatibility |
