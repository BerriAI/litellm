# Architecture decision records for the Python SDK and proxy

This folder records decisions about the Python SDK and proxy the way [`litellm-rust/docs/adrs/`](../../litellm-rust/docs/adrs/AGENTS.md) records them for the Rust core, and that file's process applies here unchanged: one short `adr_NNN_short_title.md` per decision that is hard to reverse or that a new contributor would reasonably question, with Context, Decision, Alternatives Considered, and Consequences, superseded by a new ADR instead of edited once accepted. Each folder numbers its own ADRs, so a reference to an ADR in the other folder names its path

Read the ADRs here before you build a new way to move data between the endpoints, the provider transformations, and cost tracking or logging. A PR that makes a decision of that kind adds its ADR in the same PR
