# auth_v2

A clean-slate authentication and authorization path for the proxy, built on
industry-standard libraries (authlib for JWT/OAuth2, casbin for control-plane
RBAC). It
runs in parallel with the existing auth behind a flag; when on, the existing
auth is bypassed entirely.

## Enabling

```yaml
general_settings:
  auth_version: v2
```

With the flag off, nothing changes. With it on, every request flows through the
auth_v2 entry point. Routes auth_v2 does not yet govern are left open and log a
warning ("not yet protected"); this is a work-in-progress path and must not be
enabled in production until coverage is complete.

## How it works

Authentication is a chain dispatched by credential shape:

- master key (exact, constant-time compare) -> proxy admin
- virtual key (`sk-...`) -> resolved via the existing key store
- JWT (3-part bearer) -> authlib: JWKS fetch/cache, signature + exp/iss/aud
- opaque token -> authlib RFC 7662 introspection (when configured)

Authorization splits by plane:

- control plane (management routes): a casbin engine with RBAC policy rows over
  resources `model`, `team`, `key`, `user`, `organization`, `policy`, with
  actions `read` / `write` / `delete` / `manage`. Supports per-resource ids,
  resource groups (casbin `g2`, mapping to access groups), and roles that are
  either global or scoped to a domain (casbin `g3`, e.g. "admin within
  team:eng").
- data plane (inference: chat/completions, completions, embeddings, responses):
  a plain membership/pattern predicate over the principal's allowed-model list,
  read from the already-loaded key. It is not a policy engine: on the hot path a
  casbin evaluation of a list check is pure overhead. Exact names and wildcard
  patterns (`bedrock/*`, `openai/*`, v1 semantics) match; empty list / `*` /
  `all-proxy-models` are unrestricted, matching existing key semantics.

Control-plane policies live in `LiteLLM_CasbinRule`, loaded on cold routes with a
short TTL cache; the inference path reads no policy store. A bootstrap policy
keeps `proxy_admin` fully authorized.

## Configuration

JWT (`general_settings.auth_v2_jwt` or env):

```yaml
auth_v2_jwt:
  jwks_uri: https://idp.example/.well-known/jwks.json   # AUTH_V2_JWKS_URI
  issuer: https://idp.example                            # AUTH_V2_JWT_ISSUER
  audience: litellm                                      # AUTH_V2_JWT_AUDIENCE
  user_id_claim: sub
  team_claim: team_id
  role_claim: groups
  role_map: { litellm-admins: proxy_admin }
```

OAuth2 introspection (`general_settings.auth_v2_oauth2` or env):

```yaml
auth_v2_oauth2:
  introspection_endpoint: https://idp.example/introspect  # AUTH_V2_OAUTH2_INTROSPECTION_ENDPOINT
  client_id: litellm                                      # AUTH_V2_OAUTH2_CLIENT_ID
  client_secret: ...                                      # AUTH_V2_OAUTH2_CLIENT_SECRET
  scope_claim: scope
  role_map: { "litellm:admin": proxy_admin }
```

## Policy administration

Admin-only endpoints; the policy surface governs itself.

```bash
# grant a role read access to all models
curl -sX POST localhost:4000/auth/v2/policy/permission/add \
  -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"role":"model_reader","resource":"model","action":"read"}'

# assign the role to a user (globally, or within a domain)
curl -sX POST localhost:4000/auth/v2/policy/assignment/add \
  -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"subject_type":"user","subject_id":"u_123","role":"model_reader"}'

curl -sX POST localhost:4000/auth/v2/policy/assignment/add \
  -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"subject_type":"user","subject_id":"u_123","role":"team_admin","domain":"team:eng"}'

# list all rules
curl -s localhost:4000/auth/v2/policy/list -H "Authorization: Bearer $MASTER_KEY"
```

## Live verification runbook

Prereq: generate a migration for `LiteLLM_CasbinRule`, run `make format`,
`make lint`, and `make test-unit tests/test_litellm/proxy/auth/v2/`. Start the
proxy with `auth_version: v2`.

1. master key works (bootstrap admin):
   `curl -s localhost:4000/model/info -H "Authorization: Bearer $MASTER_KEY"` -> 200
2. mint a virtual key for user `u_123` with no policy; `GET /model/info` with it
   -> 403 (governed, no grant).
3. grant `model_reader` read on models and assign to `u_123` (above); repeat
   step 2 -> 200. `POST /model/new` with that key -> 403 (read-only).
4. inference: a chat completion to a model in the key's `models` -> 200; to a
   model outside it -> 403; a key with empty `models` -> any model 200.
5. any ungoverned route still works and logs "not yet protected" in litellm.log.
