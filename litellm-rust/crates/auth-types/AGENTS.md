# auth-types

Shared vocabulary for outbound provider authentication: `Sourced<T>` and `InputSource`, credential plans and resolvers, token providers, `SecretValue`, and the `CredentialBoundary` in `src/boundary.rs`

`CredentialBoundary` is the one rule that stops a request-controlled resource, such as `api_base`, `scope`, `authority`, `project_id` or `space_id`, from being paired with a host-owned credential. Azure native auth and Azure AI OCR used to each hand-roll it. Providers choose which inputs it applies to, hosts label where each input came from, and this crate decides

```rust
let boundary = CredentialBoundary::new(&[tenant.source(), client_id.source(), secret.source()])?;
boundary.validate_resources([scope.source(), authority.source()])?;
```

`new` accepts a bundle that is entirely request-owned or entirely host-owned. `host_only` also rejects request-owned bundles. `validate_resources` rejects any request-controlled resource once the credentials are host-owned. Failures are the unit struct `UntrustedCredentialContext`, carrying no values, which each provider maps onto its existing public error

Keep this crate provider-neutral. Field names, aliases and provider messages belong to the calling provider. Before adding a credential or resource check to a provider or a host, follow [.agents/skills/rust-credential-boundary/SKILL.md](../../.agents/skills/rust-credential-boundary/SKILL.md)
