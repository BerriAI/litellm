---
name: rust-credential-boundary
description: Add or change outbound credential and resource provenance checks in litellm-rust, so a request can never steer a host-owned credential to a caller-chosen endpoint, scope, project or space, across the Python proxy, the Python SDK and the native gateway
---

# Rust credential boundary

Use this skill when a provider accepts credentials or destination-like inputs that a caller might supply per request, or when a host constructs the inputs that reach such a provider. The shared primitive is `CredentialBoundary` in `crates/auth-types/src/boundary.rs`, and that crate's `AGENTS.md` states the rule

## The problem

Every provider that lets a caller pass credentials per request has the same hole. A request pairs a resource it controls with a credential the host owns, and the proxy sends the host's secret to a destination the caller picked. Python guards this once at the proxy edge with a name blocklist in `litellm/proxy/auth/auth_utils.py`. Rust instead carries the origin of every value as `Sourced<T>` and checks the pairing right before credential acquisition, so provider-specific resources the blocklist never enumerated are covered, and the native gateway gets the same guarantee with no Python in the process

## Provider side

Pick the credential components. For a multi-part identity such as Azure tenant, client id and client secret, pass all of their sources to `CredentialBoundary::new`. For an identity a caller must never supply, such as ambient credentials, client assertions and host file references, use `host_only`

Pick the protected resources. Include the endpoint, the credential issuer or authority, the audience or scope, and the resource project or space when the provider has them. Pass every supplied spelling, including aliases and nested forms, to `validate_resources` before precedence resolution can drop one. A missing deployment value never authorizes a request override, and an equal request value is still request-controlled

Run the check before credential acquisition or any provider IO, and map `UntrustedCredentialContext` onto the provider's existing public error so accepted combinations and messages do not change. The boundary validates provenance only. URL safety, credential completeness and upstream permissions stay with the provider

Worked examples for Azure native auth, Azure AI OCR, Vertex AI OCR and Watsonx are in [references/provider-examples.md](references/provider-examples.md)

## Host side

Hosts assign `InputSource` and must not let a request forge it. Everything the request supplied is `Request`, including aliases, nested fields and connection headers. SDK callers and deployment configuration produce `Deployment`, environment lookup produces `Environment`. HTTP JSON never carries an `InputSource` or an authorization flag. A field with no label defaults to `Deployment`, so an unlabeled request field is a trusted field, and that is the bug to watch for

The existing opt-ins, `general_settings.allow_client_side_credentials` and deployment `configurable_clientside_auth_params`, are decided by the host from server-owned configuration before it constructs `Sourced` inputs. The auth crates neither read nor grant them

How the Python proxy, the Python SDK and Router, and the native gateway each fill the source map today, with request and data examples, is in [references/host-labeling.md](references/host-labeling.md)

## Tests

Cover the source cross product with `#[rstest]` and `#[values(...)]` over `InputSource::Request`, `Deployment` and `Environment` for each credential component and each protected resource, asserting accept or reject from the rule rather than from a hand-listed table. Add one named case per alias or nested spelling to show it is checked before precedence resolution. Keep the provider's existing error message assertions so the mapping stays unchanged
