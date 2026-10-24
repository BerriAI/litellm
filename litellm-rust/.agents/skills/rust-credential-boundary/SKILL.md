---
name: rust-credential-boundary
description: Add or change outbound credential and resource provenance checks in litellm-rust, or the host code that labels where request inputs came from, so a request can never steer a host-owned credential to a caller-chosen endpoint, scope, project or space
---

# Rust credential boundary

A request must never pair a resource it controls, such as `api_base`, `scope`, `project_id` or `space_id`, with a credential the host owns. Three layers decide this. Python's edge filter predates Rust, the host labels inputs on the way into Rust, and the shared boundary checks those labels

- Edge filter, in the Python proxy, name based. See Existing Python
- Labeling, in the host, records who supplied each input as `InputSource` on `Sourced<T>`
- Enforcement, in `CredentialBoundary` (`crates/auth-types/src/boundary.rs`)
  - `new` accepts a nonempty credential bundle that is all request-owned or all host-owned, and `host_only` also rejects request-owned bundles
  - `validate_resources` rejects any request-owned resource once the credentials are host-owned
  - It trusts labels completely. A wrong label passes, and a missing label means `Deployment`

## Existing Python

The proxy filters caller-controlled endpoints and identity selectors before provider dispatch. Its field-name checks do not establish provenance or authorize a Rust host to relabel an input

- `is_request_body_safe` in `litellm/proxy/auth/auth_utils.py`
  - Rejects names in `_BANNED_REQUEST_BODY_PARAMS` unless an admin opts in. Rejection becomes a 400 at the proxy auth boundary
  - The list includes `api_base`, `base_url`, `user_config`, AWS identity selectors, `vertex_credentials`, `vertex_ai_credentials`, `azure_ad_token`, endpoint-targeting fields and callback credentials and hosts
  - Checks the top-level body and one level of `_NESTED_CONFIG_KEYS` (`extra_body`, `litellm_embedding_config`) and `_NESTED_METADATA_KEYS` (`metadata`, `litellm_metadata`), including bracket-form metadata
  - Separately walks fallback targets, including `router_settings_override`, with bounded recursion in `iter_request_fallback_targets`. URL-valued fallback model names are rejected
  - `general_settings.allow_client_side_credentials: true` bypasses the banned-name loop proxy-wide. A model group's `configurable_clientside_auth_params` permits individual names, or `api_base` values matching a regex
  - These opt-ins do not bypass every check. Top-level `model_list`, server-owned federation fields and federated credential references have separate restrictions, with route-specific management exceptions
  - The per-model opt-in uses `get_model_group_info(model)` with the requested model name and a provider-prefix fallback. It does not authorize against the resolved serving deployment, so it cannot justify relabeling a resource as deployment-owned
  - A caller-supplied `api_key` does not bypass this filter. The old `check_complete_credentials` bypass is no longer used here
  - Azure `scope` and `authority`, and Vertex `project` and `location`, are not banned by those names. The filter is not a general resource-ownership check
- `litellm/router_utils/clientside_credential_handler.py`
  - Copies caller `api_key`, `api_base` and `base_url` onto the deployment's `litellm_params`
  - On a caller base override, rejects an effective `api_key` that is an `oidc/` secret reference. A caller-supplied key has already replaced the deployment key at this point
  - Clears `_ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE`, including organization, `extra_body`, region, token, Vertex and AWS fields, then restores fields the caller supplied. It retains the effective `api_key`
  - Preserves per-user Copilot credential selection when locked and disables Anthropic workload identity federation on redirected calls
- What reaches Rust
  - `request_input_sources` in `crates/python-bridge/src/marshal.rs` reads `proxy_server_request.body_fields` (or `body`) and `credential_fields`. Among the names the route asks it to inspect, it labels matching names `Request`
  - It never reads the admin opt-ins or recursively discovers nested aliases. The host must supply all relevant names and propagate their labels
  - An allowed caller `api_base` therefore remains `Request`. When a route enforces `CredentialBoundary`, host credentials still reject it. Fully request-owned credentials can pass `new`, but not a provider's `host_only` or stricter check
  - This is the intended rule for pure Rust routes. For a route with a Python fallback, record any parity divergence instead of hiding it by relabeling
- What Python does not give you
  - The name-based filter and config merge do not create Rust ownership labels
  - The banned-name list covers some non-endpoint selectors, but is not exhaustive. Providers must validate protected resources such as scope, authority, project and space
  - Python-only routes still need their edge filter, stripping and provider-specific checks until they move to Rust

## Providers

- Pass every credential component to `CredentialBoundary::new`
  - Use `host_only` for ambient identities, client assertions and host file references
- Pass every protected resource to `validate_resources`
  - The endpoint, issuer or authority, audience or scope, and project or space
  - Every alias and nested spelling, before precedence resolution drops one. An equal request value is still request-owned
- Run the check before credential acquisition or any provider IO
- Map `UntrustedCredentialContext` onto the provider's existing public error, so accepted combinations and messages stay the same
- References are `crates/auth-azure/src/native.rs` and `crates/llms/src/azure_ai/common_utils.rs`
  - Vertex OCR deliberately keeps its stricter standalone `validate_destination` in `crates/llms/src/vertex_ai/ocr/common_utils.rs`. It rejects any supplied `Request` `api_base`, even with request-owned credentials. This PR does not loosen that policy

## Hosts

- Every field the caller sent is `Request`, including nested fields, aliases and connection headers
  - HTTP input never carries a label or an authorization flag
  - Deployment config and in-process SDK callers are `Deployment`, and environment lookups are `Environment`
- Relabeling a caller field as `Deployment` because an admin opt-in allows it is the risky step
  - See Existing Python for why the edge opt-in is not serving-deployment authorization. The boundary cannot detect a wrong label
- What to do depends on the route
  - Pure Rust with no Python fallback, such as tokenize
    - Never relabel. Caller fields are always `Request`
    - Proxy calls still pass through the edge filter. SDK calls must not rely on it
    - A caller who wants to override a resource sends a full credential set
  - Rust with a Python fallback that must keep parity
    - If Python honors an opt-in, enforcing request provenance in Rust can diverge
    - Any compatibility authorization must use the resolved serving deployment's config. Otherwise preserve `Request` and record the divergence
  - Python only
    - Keep the edge filter, field stripping and provider-specific checks until the route moves to Rust

## Tests

- Cover the `InputSource` cross product for each credential component and resource with `#[rstest]` and `#[values(...)]`, asserting the rule instead of a hand-listed table
- Add one named case per alias or nested spelling
- Keep the provider's existing error message assertions
- Follow-up for the tokenize host layer, which is not added here
  - With the first host that labels tokenize inputs, add a test that a caller field stays `Request` even when the deployment lists it in `configurable_clientside_auth_params`
  - Add the test's file path here when it lands. This PR does not change host labeling or claim that test exists
