# auth-types

Shared vocabulary for outbound provider authentication: `Sourced<T>` and `InputSource`, credential plans and resolvers, token providers, `SecretValue`, and the `CredentialBoundary` in `src/boundary.rs`

`CredentialBoundary` stops a request-controlled resource from being paired with a host-owned credential. Keep this crate provider-neutral, so field names, aliases and messages belong to the calling provider. Before adding a credential or resource check to a provider, or labeling inputs in a host, follow [.agents/skills/rust-credential-boundary/SKILL.md](../../.agents/skills/rust-credential-boundary/SKILL.md)
