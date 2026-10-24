---
name: rust-credential-boundary
description: Add or change outbound credential and resource provenance checks in litellm-rust, or the host code that labels where request inputs came from, so a request can never steer a host-owned credential to a caller-chosen endpoint, scope, project or space
---

# Rust credential boundary

A request must never pair a resource it controls, such as `api_base`, `scope`, `project_id` or `space_id`, with a credential the host owns. Three layers decide this, and only the last one is Rust

- Edge filter, in the Python proxy for every route
  - `is_request_body_safe` in `litellm/proxy/auth/auth_utils.py` rejects caller fields by name, such as `api_base` and `base_url`
  - `general_settings.allow_client_side_credentials` or a model group's `configurable_clientside_auth_params` let them through, checked against the model name the caller sent, before alias resolution
  - When a caller `api_base` is allowed, `litellm/router_utils/clientside_credential_handler.py` strips the admin's provider config but keeps `api_key`, because the admin vouched for that destination
- Labeling, in the host
  - The host records who supplied each input as `InputSource` on `Sourced<T>`
  - Only routes that reach Rust have labels. Python-only routes have none
- Enforcement, in `CredentialBoundary` (`crates/auth-types/src/boundary.rs`)
  - `new` accepts a credential bundle that is all request-owned or all host-owned, and `host_only` also rejects request-owned bundles
  - `validate_resources` rejects any request-owned resource once the credentials are host-owned
  - It trusts labels completely. A wrong label passes, and a missing label means `Deployment`

## Providers

- Pass every credential component to `CredentialBoundary::new`
  - Use `host_only` for ambient identities, client assertions and host file references
- Pass every protected resource to `validate_resources`
  - The endpoint, issuer or authority, audience or scope, and project or space
  - Every alias and nested spelling, before precedence resolution drops one. An equal request value is still request-owned
- Run the check before credential acquisition or any provider IO
- Map `UntrustedCredentialContext` onto the provider's existing public error, so accepted combinations and messages stay the same
- References are `crates/auth-azure/src/native.rs` and `crates/llms/src/azure_ai/common_utils.rs`

## Hosts

- Every field the caller sent is `Request`, including nested fields, aliases and connection headers
  - HTTP input never carries a label or an authorization flag
  - Deployment config and in-process SDK callers are `Deployment`, and environment lookups are `Environment`
- Relabeling a caller field as `Deployment` because an admin opt-in allows it is the risky step
  - It expresses "the admin trusts where this sends the key", which is legitimate only when decided against the deployment that actually serves the request
  - Deciding it from the requested model name before alias resolution lets an allowed alias borrow a protected deployment's credentials, and the boundary cannot see the wrong label
- What to do depends on the route
  - Pure Rust with no Python fallback, such as tokenize
    - Never relabel. Caller fields are always `Request`
    - The edge filter still runs and is harmless redundancy
    - A caller who wants to override a resource sends a full credential set
  - Rust with a Python fallback that must keep parity
    - If the Python path honors an opt-in, the Rust labeling has to as well, or the paths diverge
    - Decide it after deployment resolution from that deployment's config, or record the divergence
  - Python only
    - The edge filter and the field stripping are the only fence. They are name-based and inconsistent, but keep them until the route moves to Rust

## Tests

- Cover the `InputSource` cross product for each credential component and resource with `#[rstest]` and `#[values(...)]`, asserting the rule instead of a hand-listed table
- Add one named case per alias or nested spelling
- Keep the provider's existing error message assertions
- On the host side, assert that a caller field stays `Request` even when the deployment lists it in `configurable_clientside_auth_params`
