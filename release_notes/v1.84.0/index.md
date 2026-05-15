---
title: "v1.84.0 - Reliability hardening + multi-pod budget accuracy"
slug: "v1-84-0"
date: 2026-05-14T00:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Yuneng Jiang
    title: Senior Full Stack Engineer, LiteLLM
    url: https://www.linkedin.com/in/yuneng-david-jiang-455676139/
    image_url: https://avatars.githubusercontent.com/u/171294688?v=4
hide_table_of_contents: false
---

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

```bash
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
docker.litellm.ai/berriai/litellm:1.84.0
```

</TabItem>
<TabItem value="pip" label="Pip">

```bash
pip install litellm==1.84.0
```

</TabItem>
</Tabs>

## Version naming change

> **Starting with `v1.84.0`, LiteLLM versions follow [PEP 440](https://peps.python.org/pep-0440/).** Stable releases drop the `-stable` suffix — the Docker tag for this release is `litellm:1.84.0`, not `litellm:1.84.0-stable`. Every Docker tag is published in both bare and `v`-prefixed form (`litellm:1.84.0` and `litellm:v1.84.0` resolve to the same image), so existing pins that include the `v` prefix keep working. PyPI versions remain the bare PEP 440 form: `pip install litellm==1.84.0`. If you pin LiteLLM in deployment tooling (Helm values, `requirements.txt`, Renovate rules, etc.), update those pins to the PEP 440 form.

Mapping from the legacy suffix scheme to the new PEP 440 scheme:

| Channel | Legacy (≤ `v1.83.x`) | New (≥ `v1.84.0`) |
| --- | --- | --- |
| Stable | `vX.Y.Z-stable` | `vX.Y.Z` |
| Stable patch | `vX.Y.Z-stable.patch.N` | `vX.Y.Z.postN` |
| Release candidate | `vX.Y.Z.rc.N` / `vX.Y.Z-rc.N` | `vX.Y.ZrcN` |
| Dev / nightly | `vX.Y.Z-nightly` / `vX.Y.Z.dev.N` | `vX.Y.Z.devN` |

This is a naming change only — release cadence, stability guarantees, and image contents are unchanged. The `v1.84.0-rc.1` tag (cut before the switch) keeps the legacy form for historical continuity; every tag from `v1.84.0` onward uses the PEP 440 form.

---

> **Heads up — large bundle of behavioral changes.** This release consolidates a lot of reliability and hardening work that shipped in tight sequence. The **Important Behavior Changes** section below covers everything that changes a default, removes a configuration shortcut, or alters a request/response shape, with the opt-out you need to keep prior behavior. Read that section before upgrading a production deployment. If you already validated against `v1.84.0-rc.1`, see the **Changes since v1.84.0-rc.1** section for the post-rc delta.

## Key Highlights

- **Pass-through endpoints are authenticated by default.** The `auth` field on entries under `general_settings.pass_through_endpoints` now defaults to `true`. The previous "OSS gets unauthenticated forwarders by default; `auth: true` is enterprise-only" combination is gone — `auth: true` works on OSS, and operators who want an unauthenticated forwarder must set `auth: false` explicitly.
- **Multi-pod budget enforcement is materially more accurate.** `RedisCache.async_increment` gains a `refresh_ttl` opt-in, spend counters opt into it, and stale in-memory counters are skipped on a clean Redis miss. `ResetBudgetJob` invalidates Redis counters alongside DB resets so refreshed counters get reset too.
- **Prisma DB reconnects no longer freeze the event loop.** The reconnect path replaced `await self.db.disconnect()` (which called `subprocess.Popen.wait()` synchronously) with a SIGTERM→SIGKILL → fresh `Prisma()`+`connect()` sequence. Liveness probes stop failing during database flaps. Companion fix restores reconnect-and-retry on `PrismaClient.get_generic_data`.
- **Memory footprint down ~700 MB** on a two-worker Docker deployment via lazy-loaded feature routers and lazy-loaded front page. First request to a lazy route incurs the import cost; subsequent requests are unchanged.
- **MCP OAuth + Azure Entra discovery support**, opt-in short-ID tool prefix to keep MCP tool names under the 60-char limit, and OAuth root-endpoint visibility now matches explicit server-name lookup.
- **Durable agent workflow run tracking** via a new `/v1/workflows/runs` REST surface backed by `LiteLLM_WorkflowRun` / `LiteLLM_WorkflowEvent` / `LiteLLM_WorkflowMessage` tables. Spend logs `session_id` joins for free cost attribution.
- **Per-model routing strategies via Routing Groups.** New `router_settings.routing_groups` schema binds a list of `model_name`s to its own routing strategy (e.g. `latency-based-routing` for `gpt-4o`, `simple-shuffle` for cheaper models) within a single router. Configurable in `proxy_config.yaml` or from the LiteLLM dashboard under General Settings → Routing Groups; UI-managed groups persist and override the YAML values.

---

## Changes since `v1.84.0-rc.1`

Everything below landed on top of `v1.84.0-rc.1` and is included in `v1.84.0`. If you already validated against the rc, this is the only delta to re-test.

### Hardening
- **`/key/update` authorization checks** — [PR #27878](https://github.com/BerriAI/litellm/pull/27878)
- **`/key/regenerate` ownership-rebind + premium-gate guards** — [PR #27793](https://github.com/BerriAI/litellm/pull/27793)
- **Reject bare strings at file-input sinks** to prevent local-file reads via crafted request bodies — [PR #27762](https://github.com/BerriAI/litellm/pull/27762)
- **Refuse remote-URL instance-fn loads** outside the config-file path — [PR #27801](https://github.com/BerriAI/litellm/pull/27801)
- **Cover `extra_body` + `azure_ad_token` in banned-params check** — [PR #27898](https://github.com/BerriAI/litellm/pull/27898)
- **MCP BYOK / OAuth: block SSRF fields in RAG ingest `vector_store` config; block client-side pricing injection via request body** — [PR #27892](https://github.com/BerriAI/litellm/pull/27892)

### Budget reservation
- **Bound budget reservation per request** instead of pinning to the entire remaining team/key/user headroom on requests without `max_tokens` — [PR #27509](https://github.com/BerriAI/litellm/pull/27509)
- **Image generation: reserve per-image cost** rather than max-tokens cost; gate strictly on model mode

### Health probes
- **Re-expose `db` status on the unauthenticated `/health/readiness` payload** so external probes can distinguish DB-unreachable workers without auth — [PR #27866](https://github.com/BerriAI/litellm/pull/27866)
- **UI fetches `litellm_version` + `is_detailed_debug` from `/health/readiness/details`** (auth-gated) since those fields were moved off the public payload — [PR #27896](https://github.com/BerriAI/litellm/pull/27896)
- **UI: disable retries on `/health/readiness/details` + cover token forwarding**

### MCP
- **Forward configured `extra_headers` from the MCP client to upstream OpenAPI HTTP calls** (closes [#26794](https://github.com/BerriAI/litellm/issues/26794)) — [PR #27383](https://github.com/BerriAI/litellm/pull/27383)
- **On the same forwarding path, `static_headers` now win over caller-forwarded `extra_headers` on name conflict** (case-insensitive). See [Important Behavior Changes → MCP](#openapi-mcp-static_headers-now-win-over-caller-forwarded-extra_headers) below.

### Routing under `SERVER_ROOT_PATH`
- **Lazy-feature loading under a non-empty `SERVER_ROOT_PATH`** no longer 404s on routes such as `/api/v1/policies/attachments/list`; strip the prefix before lazy-feature match and cache the normalized path at middleware init — [PR #27812](https://github.com/BerriAI/litellm/pull/27812)

### Tagging & metrics
- **⚠️ Reverted the v1.83.10 caller-tag strip / `allow_client_tags` opt-in** — caller-supplied tags merge into request metadata again; the strip is no longer enforced. **See the new entry under Important Behavior Changes → Tags below for the full impact.** — [PR #27789](https://github.com/BerriAI/litellm/pull/27789)
- **Point the `/metrics` 401 hint at the actual opt-out flag** — [PR #27505](https://github.com/BerriAI/litellm/pull/27505)

### Packaging
- **Relax core runtime pins to ranges** so downstream packages can resolve a single shared `openai`/etc. version — [PR #27241](https://github.com/BerriAI/litellm/pull/27241)
- **Raise `jinja2` floor in `[project.dependencies]` to `>=3.1.6`** to match the lockfile — [PR #27552](https://github.com/BerriAI/litellm/pull/27552)

---

## ⚠️ Important Behavior Changes

This release tightens a number of defaults across auth, ingress, callbacks, MCP, and the UI. Each item below names the change and, where applicable, the exact configuration you need to restore prior behavior.

### Auth & request ingress

#### Pass-through endpoints default to `auth: true`
- **What changed:** `PassThroughGenericEndpoint.auth` now defaults to `True`. The runtime dispatch in `user_api_key_auth.py` reads endpoints as raw dicts, so `endpoint.get("auth", True)` applies even when the dict has no explicit key. The `premium_user` gate on `auth: true` was also removed — OSS deployments can now use `auth: true`.
- **Who is affected:** Any pass-through entry in `general_settings.pass_through_endpoints` that omitted `auth:`. Prior to this rc that meant unauthenticated; it now means LiteLLM-key-authenticated.
- **Restore prior behavior:** Set `auth: false` explicitly on every pass-through entry that is meant to be public (e.g. webhook receivers).
  ```yaml
  general_settings:
    pass_through_endpoints:
      - path: /webhook/something
        target: https://example.com/webhook
        auth: false   # was implicit before; must be explicit now
  ```

#### Clientside `api_base` / `base_url` are gated and credential-stripped
- **What changed:**
  1. Clientside `api_base` / `base_url` are validated against `validate_url` when `litellm.user_url_validation` is enabled.
  2. When a request redirects `api_base` / `base_url`, admin-configured provider credentials and per-deployment metadata (OCI signing keys, AWS / Azure / Vertex tokens, observability vars, every field on `CredentialLiteLLMParams`) are dropped before the call is forwarded.
  3. The provider-inference matcher in `get_llm_provider_logic.py` no longer does an unanchored substring match — it now compares parsed URL hostname + segment-bounded path prefix.
  4. The blocklist for clientside-overridable params adds `aws_bedrock_runtime_endpoint`, `langsmith_base_url`, `langfuse_host`, `posthog_host`, `braintrust_host`, `slack_webhook_url`, `s3_endpoint_url`, `sagemaker_base_url`, `deployment_url`. The old "blocklist is a no-op when `api_key` is non-empty" clause is removed.
- **Who is affected:** Anyone passing `api_base` (or any of the newly-blocked fields) at request time and relying on the implicit-`api_key` bypass to thread it through.
- **Restore prior behavior:** Use the documented BYOK paths instead of the bypass:
  - Proxy-wide: `general_settings.allow_client_side_credentials: true`
  - Per deployment: `litellm_params.configurable_clientside_auth_params: ["api_base", ...]`

  The 400 returned by the proxy on a blocked request names the offending field and points at the same two settings.

#### Master-key requests now propagate an alias instead of the master-key hash
- **What changed:** When a request authenticates with the master key, the `UserAPIKeyAuth.api_key` / `token` value handed to downstream code is now the constant `LITELLM_PROXY_MASTER_KEY_ALIAS = "litellm_proxy_master_key"`. The cache lookup is unchanged (still keyed on `hash_token(master_key)`). `_is_master_key` no longer accepts the SHA-256 hash form — only the raw master key.
- **Who is affected:** Anything joining or filtering on the prior master-key hash value, including custom dashboards over spend logs and Prometheus `/metrics` queries pinned to the hash literal.
- **Restore prior behavior:** None — operators querying spend logs or metrics for master-key activity should switch their filter to the alias `"litellm_proxy_master_key"`.

#### Invite-link onboarding no longer mints a key from `GET`
- **What changed:** `GET /onboarding/get_token` returns a 15-minute signed onboarding JWT bound to invite + user id; it does **not** mint a `sk-...` virtual key. `POST /onboarding/claim_token` requires that JWT and atomically reserves the invite via `update_many(... is_accepted=False, ... → True)`.
- **Who is affected:** Any tooling that consumed `GET /onboarding/get_token` for an embedded `sk-...` and treated it as a usable session key before completing the password claim.
- **Restore prior behavior:** None — clients must call `POST /onboarding/claim_token` to obtain the live key.

#### CLI SSO login flow uses a server-side session
- **What changed:** `litellm-proxy login` now starts a CLI SSO flow that returns a login id + polling secret + terminal verification code. The browser callback must confirm the terminal code before the polling endpoint returns the JWT.
- **Who is affected:** Anyone running an older `litellm-proxy` CLI against an upgraded proxy — the old caller-supplied-handle handoff is gone.
- **Restore prior behavior:** None — upgrade the CLI alongside the proxy.

#### Team self-join (`_is_available_team`) only allows self-add as `role=user`
- **What changed:**
  - `/team/member_add`: when the caller is not an admin and the team is "available," the request must add **only the caller themselves** with **`role="user"`**. Bulk shapes are checked the same way; lists mixing a valid self-entry with a `role="admin"` entry are rejected. Email-only members on the self-join path are rejected.
  - `/team/permissions_update`: the `_is_available_team` clause is removed entirely — only proxy/team/org admins can update `team_member_permissions`.
- **Who is affected:** Any flow that relied on the blanket bypass to either add an admin to an available team without admin privileges, or to mutate `team_member_permissions` from a non-admin context.
- **Restore prior behavior:** None — perform admin-scoped operations with an admin key.

#### Guardrail modification permission gates on key presence
- **What changed:** The guardrail-modification authz check in `auth_checks.py` now gates on intent (whether the key is present in the request) rather than payload truthiness. Some previously-accepted shapes will now 403.
- **Restore prior behavior:** None — flow updates required for non-admin callers that previously slipped past on falsy payloads.

#### Untrusted root control fields are stripped from client requests
- **What changed:** `_UNTRUSTED_ROOT_CONTROL_FIELDS` in `litellm_pre_call_utils.py` includes `mock_response`, `mock_tool_calls`, redaction-bypass controls, and a few others. They are stripped from client requests unless the calling key/team carries `allow_client_mock_response: true` (for `mock_response` / `mock_tool_calls`) or the corresponding admin-opt-in metadata for the redaction bypass. Pillar guardrail caching headers and Bedrock dynamic evaluation overrides are also filtered when not explicitly allowed.
- **Who is affected:** Tests and tooling that pass `mock_response` / `mock_tool_calls` in `extra_body` to short-circuit completions.
- **Restore prior behavior:** Set `allow_client_mock_response: true` in the admin metadata of the test key (or the team owning it):
  ```python
  client.keys.generate(
      key_alias="ci-mock-key",
      metadata={"allow_client_mock_response": True},
  )
  ```

#### Error responses no longer leak re-raised local parameters
- **What changed:** Broad `except` handlers in the response-utils path used to render the captured request parameters into the re-raised error message. Those parameters can carry credentials, so they're now dropped from the rendered message.
- **Who is affected:** Any client that parsed credential-shaped fields out of a 5xx error body. The error response shape is otherwise unchanged.
- **Restore prior behavior:** None.

### Vector stores

#### Credentials redacted; `/vector_store/update` is per-store gated
- **What changed:**
  - `/vector_store/list`, `/vector_store/info`, `/vector_store/update` redact credential-bearing values inside the persisted `litellm_params` (handles dicts, JSON-string-serialized params, and nested-dict shapes like `litellm_embedding_config`).
  - `/vector_store/update` is now gated by `_fetch_and_authorize_vector_store` — same per-store access check `/vector_store/info` already had.
  - `SensitiveDataMasker` adds plural `"credentials"` to its default sensitive-pattern set, so segment-exact matching catches `vertex_credentials`, `aws_credentials`, etc. (Latent fix that affects every default-instantiated masker, not just vector stores.)
  - `get_vector_store_info` and `update_vector_store` re-raise `HTTPException` instead of letting the catch-all downgrade `403` / `404` to `500`.
- **Who is affected:** Anything reading `litellm_params` off these responses to recover provider keys, or any non-store-admin caller mutating arbitrary vector stores via `/vector_store/update`.
- **Restore prior behavior:** None.

### Logging callbacks & key/team metadata

#### `os.environ/*` callback refs in key/team metadata are no longer resolved
- **What changed:** `convert_key_logging_metadata_to_callback()` no longer resolves `os.environ/*` values from key/team metadata via `get_secret()`. Existing rows with such values are silently ignored at request setup instead of crashing the request. Trusted `config.yaml` team-callback env resolution in `add_team_based_callbacks_from_config()` is unchanged. New `AddTeamCallback` constructions from key/team logging metadata also reject `os.environ/*` callback vars.
- **Who is affected:** Any key/team that stored `os.environ/DATABASE_URL` (or similar) in its callback metadata to pick up a server env var at request time.
- **Restore prior behavior:** Configure those callback secrets through trusted proxy `config.yaml` (`team_callbacks` / `model_list[*].litellm_params`) instead of putting `os.environ/*` references in DB-backed key or team metadata. The literal credential value can still be stored in metadata if absolutely necessary.

#### Team-callback admin mutations now emit audit logs
- **What changed:** `POST /team/{id}/callback` (`add_team_callbacks`) and `POST /team/{id}/disable_logging` (`disable_team_logging`) emit `LiteLLM_AuditLogs` rows when `litellm.store_audit_logs=True`. Additive when audit logging is enabled.
- **Restore prior behavior:** `litellm.store_audit_logs: false` (the default) suppresses the new rows.

### MCP

#### Encrypted user-scoped MCP credentials at rest
- **What changed:** Writes to `LiteLLM_MCPUserCredentials.credential_b64` go through `encrypt_value_helper` (nacl SecretBox) instead of plain `urlsafe_b64encode`. The read path tries nacl decryption first and falls back to plain `urlsafe_b64decode` for legacy rows; existing rows stay readable.
- **Who is affected:** Operators reading the table directly; the column contents change shape on first re-write.
- **Restore prior behavior:** None — backward-compat read path keeps legacy rows working until they are next written.

#### OAuth metadata discovery follows SSRF guard
- **What changed:** The two URLs MCP discovery follows (`resource_metadata` from `WWW-Authenticate`, and `authorization_servers[0]` from protected-resource-metadata) are now subject to `async_safe_get`. Same-authority metadata fetches stay direct (with `follow_redirects=False`); cross-origin fetches are validated via the existing user URL validation policy. Public federated providers (Azure Entra, Google, Okta, GitHub) remain supported.
- **Who is affected:** Cross-origin internal/loopback/cloud-metadata OAuth metadata URLs.
- **Restore prior behavior:** Toggle `litellm.user_url_validation` and the existing URL validation controls per the proxy URL-validation docs to permit your specific internal targets.

#### MCP public-route detection no longer matches query strings; OAuth2 fallback no longer fail-opens
- **What changed:**
  - `MCPRequestHandler.process_mcp_request` checks `request.url.path.startswith("/.well-known/")` instead of `".well-known" in str(request.url)`. Query-string smuggling like `?.well-known` is rejected.
  - When an `Authorization` header fails LiteLLM-key validation, the handler no longer treats the failure as "OAuth2 passthrough" and returns an empty `UserAPIKeyAuth()`.
- **Restore prior behavior:** None.

#### MCP OAuth root endpoint resolves with request visibility rules
- **What changed:** Root-endpoint fallback resolves the single OAuth2 server using the same visibility rules as explicit server-name lookup; non-visible servers are no longer selected via the fallback path. The callback redirect path validates the full client redirect URI carried in state and appends parameters without dropping an existing query string.
- **Restore prior behavior:** None — adjust server visibility rather than relying on the fallback.

#### OpenAPI MCP: `static_headers` now win over caller-forwarded `extra_headers`
- **What changed:** v1.84.0 introduced header forwarding for OpenAPI-backed MCP servers (`spec_path:` configs) via [PR #27383](https://github.com/BerriAI/litellm/pull/27383), letting you allowlist caller request headers into upstream OpenAPI HTTP calls. When the same header name appears in both your YAML `static_headers` and the request-time `extra_headers` allowlist, the **`static_headers` value now wins**, with case-insensitive name comparison so `X-Tenant-Id` and `x-tenant-id` are treated as the same header. This matches how the managed MCP path has always behaved. `Authorization` is still overridden last by a BYOK `x-mcp-auth` token, if present.
- **Example:** With
  ```yaml
  mcp_servers:
    data_api:
      spec_path: http://upstream-api.local/openapi.json
      static_headers:
        X-Tenant-Id: "acme-corp"
      extra_headers:
        - X-Tenant-Id
  ```
  a caller sending `X-Tenant-Id: evil-corp` will now have `X-Tenant-Id: acme-corp` sent upstream. Any header in `extra_headers` that does **not** collide with `static_headers` is still forwarded unchanged.
- **Who is affected:** Operators who set the same header name in both `static_headers` and `extra_headers` on an OpenAPI MCP server, and who were relying on the caller's value taking effect. (Note: this only ever shipped in the v1.84.0 release-candidate cycle — no prior stable release forwarded `extra_headers` for OpenAPI MCPs at all.)
- **Restore prior behavior:** None — if you actually want the caller to control a header, remove it from `static_headers` and keep it only in `extra_headers`, or use distinct names for the operator-pinned value and the caller-supplied value.

### UI / static assets

#### `/get_image`, `/get_favicon`, `/get_logo_url`
- **What changed:**
  - Remote HTTP(S) `UI_LOGO_PATH` / `LITELLM_FAVICON_URL` are now browser-loaded via redirect — the proxy no longer fetches them server-side from these unauthenticated endpoints.
  - Local file paths still work in place, but the resolved file must have a supported image signature (`jpeg`, `png`, `gif`, `webp`, `ico`); non-image paths fall back to the bundled default.
  - `/get_logo_url` only returns HTTP(S) values; local filesystem paths are not disclosed.
  - Stale `cached_logo.jpg` files are no longer served by `/get_image`.
- **Who is affected:** Custom branding setups that pointed `UI_LOGO_PATH` / `LITELLM_FAVICON_URL` at non-image local files, or relied on `/get_logo_url` to surface a local path.
- **Restore prior behavior:** No new env vars required. Existing remote URLs continue to work; local image paths continue to work as long as the file is a recognized image type.

#### `/ui/chat` removed
- **What changed:** Static `chat.html` / `chat.txt` / `chat/` are gone; the route 404s. The chat UI was already removed from the nav; the dangling static build is now also gone.
- **Restore prior behavior:** None.

#### "Store Prompts in Spend Logs" toggle moved to Admin Settings
- **What changed:** Both "Store Prompts in Spend Logs" and "Maximum Spend Logs Retention Period" moved from a gear-icon modal on the Logs page to **Admin Settings → Logging Settings**. The gear was visible to non-admins and surfaced 403s on save.
- **Restore prior behavior:** None — controls are admin-only as `/config/update` and `/config/list` already required.

### Tags

#### ⚠️ Reverted: v1.83.10 caller-tag strip / `allow_client_tags` opt-in
- **What changed:** **This release reverts the [v1.83.10 breaking change](/release_notes/v1.83.10/v1-83-10) that stripped caller-supplied tags unless the key/team metadata had `allow_client_tags: true`.** Caller-supplied tags from `x-litellm-tags`, body-level `tags`, and `metadata.tags` now flow into `metadata.tags` again and union with admin-configured static tags from key/team/project metadata — the proxy's behavior is back to what it was before v1.83.10. The pre-call strip block in `litellm_pre_call_utils.py` is removed, and the flag has no schema or endpoint footprint, so leftover `allow_client_tags: true` values on existing keys/teams are inert.
- **Who is affected:**
  - Operators who set `metadata.allow_client_tags: true` on keys/teams to opt into client tags: the flag is now a no-op and can be cleaned up at leisure.
  - **Operators who relied on the v1.83.10 strip to block client-supplied tags reaching tag-based routing or tag-based spend attribution: the strip is no longer enforced.** Re-evaluate your tag-based routing and cost-attribution exposure before upgrading.
- **Restore prior behavior:** None — the strip path is gone from the proxy. If caller-supplied tags must be blocked, filter them upstream (gateway / ingress) or in a custom pre-call hook.

---

## New Models / Updated Models

#### New Model Support (16 new models)

| Provider     | Model                                          | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features                                                  |
| ------------ | ---------------------------------------------- | -------------- | ------------------- | -------------------- | --------------------------------------------------------- |
| OpenAI       | `gpt-image-2`, `gpt-image-2-2026-04-21`        | n/a (image)    | $5.00               | $10.00               | vision, pdf input                                         |
| Azure OpenAI | `azure/gpt-image-2`, `azure/gpt-image-2-2026-04-21` | n/a (image) | $5.00               | $10.00               | vision, pdf input                                         |
| AWS Bedrock  | `zai.glm-5`                                    | 200,000        | $1.00               | $3.20                | function calling, reasoning, tool choice                  |
| Crusoe       | `crusoe/deepseek-ai/DeepSeek-R1-0528`          | 163,840        | $3.00               | $7.00                | reasoning                                                 |
| Crusoe       | `crusoe/deepseek-ai/DeepSeek-V3-0324`          | -              | -                   | -                    | -                                                         |
| Crusoe       | `crusoe/google/gemma-3-12b-it`                 | 131,072        | $0.10               | $0.10                | function calling, vision, tool choice                     |
| Crusoe       | `crusoe/meta-llama/Llama-3.3-70B-Instruct`     | 131,072        | $0.20               | $0.20                | function calling, tool choice                             |
| Crusoe       | `crusoe/moonshotai/Kimi-K2-Thinking`           | 262,144        | $2.50               | $2.50                | reasoning                                                 |
| Crusoe       | `crusoe/openai/gpt-oss-120b`                   | 131,072        | $0.80               | $0.80                | function calling, tool choice                             |
| Crusoe       | `crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507`    | 262,144        | $3.00               | $3.00                | function calling, tool choice                             |
| Vertex AI    | `vertex_ai/xai/grok-4.1-fast-reasoning`        | 2,000,000      | $0.20               | $0.50                | function calling, vision, reasoning, response schema, tool choice |
| Vertex AI    | `vertex_ai/xai/grok-4.1-fast-non-reasoning`    | 2,000,000      | $0.20               | $0.50                | function calling, vision, response schema, tool choice    |
| Vertex AI    | `vertex_ai/xai/grok-4.20-reasoning`            | 2,000,000      | $2.00               | $6.00                | function calling, vision, reasoning, response schema, tool choice |
| Vertex AI    | `vertex_ai/xai/grok-4.20-non-reasoning`        | 2,000,000      | $2.00               | $6.00                | function calling, vision, response schema, tool choice    |

#### New Providers (2 new providers)

| Provider     | Endpoints                                              | Notes                                                                                |
| ------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **AIHubMix** | OpenAI-compatible chat completions                     | [PR #24294](https://github.com/BerriAI/litellm/pull/24294)                            |
| **Crusoe**   | chat completions across reasoning / instruct catalogs  | catalog above                                                                        |

#### Pricing updates

- **OpenAI [`gpt-5.5-pro`](../../docs/providers/openai)** — corrected: was 2× OpenAI's published rate. Cost-tracking output for `gpt-5.5-pro` will drop to half what it reported under previous releases — operators reconciling spend reports across the upgrade boundary should expect the discontinuity. - [PR #26651](https://github.com/BerriAI/litellm/pull/26651)
- **AWS Bedrock Anthropic Claude 4.5 / 4.6 / 4.7** (Global + US) — added `cache_creation_input_token_cost_above_1hr` (and the `_above_200k_tokens` LC variant for Sonnet 4.5). 1-hour-TTL prompt-cache writes on Bedrock now bill at the published 1.6× rate instead of falling back to the 5-minute rate (was undercounting by ~60%). - [PR #26800](https://github.com/BerriAI/litellm/pull/26800)

#### Features

- **[Bedrock](../../docs/providers/bedrock)**
    - Preserve `cache_control` TTL on tools for Claude 4.5+ on the Converse path; sanitize `tools` blocks on the Invoke path - [PR #25855](https://github.com/BerriAI/litellm/pull/25855)
    - Translate OpenAI `file` content on the tool-result path (Bedrock Converse + direct Anthropic) - [PR #26710](https://github.com/BerriAI/litellm/pull/26710)
    - `retrievalConfiguration` passthrough for vector-store search via `extra_body` - [PR #26685](https://github.com/BerriAI/litellm/pull/26685)
- **[Vertex AI](../../docs/providers/vertex)**
    - Propagate metadata labels to embeddings (`labels`), Imagen (`labels`), and Discovery Engine rerank (`userLabels`); shared helper across paths - [PR #25499](https://github.com/BerriAI/litellm/pull/25499)
    - Reuse Anthropic-messages config instances via `@lru_cache` so `VertexBase` credential cache survives across calls - [PR #26099](https://github.com/BerriAI/litellm/pull/26099)
- **[Google Native](../../docs/pass_through/google_ai_studio)**
    - Emit LiteLLM proxy success headers (`x-litellm-*`) on `:generateContent` and `:streamGenerateContent` - [PR #25500](https://github.com/BerriAI/litellm/pull/25500)
    - Run `pre_call_hook` on `:generateContent` / `:streamGenerateContent` so guardrails fire - [PR #26914](https://github.com/BerriAI/litellm/pull/26914)
- **[Anthropic](../../docs/providers/anthropic)**
    - JSON `response_format` + user tools on non-streaming: filtered tool calls + structured JSON merged into `content`; internal `json_tool_call` no longer surfaces - [PR #26222](https://github.com/BerriAI/litellm/pull/26222)
- **[Ollama](../../docs/providers/ollama)**
    - Forward `tool_calls` on assistant messages and `tool_call_id` on `role: tool` messages — fixes the infinite tool-call loop on multi-turn agents - [PR #26122](https://github.com/BerriAI/litellm/pull/26122)
- **[Predibase](../../docs/providers/predibase)**
    - Migrate `transform_request` / `transform_response` into `transformation.py` (refactor, no behavior change) - [PR #25249](https://github.com/BerriAI/litellm/pull/25249)
- **[AIHubMix](../../docs/providers/aihubmix) (new)**
    - First-class OpenAI-compatible provider entry - [PR #24294](https://github.com/BerriAI/litellm/pull/24294)

### Bug Fixes

- **[Vertex AI](../../docs/providers/vertex)**
    - Preserve `items` on the array branch of `anyOf` schemas with `null` (Vertex was rejecting `INVALID_ARGUMENT`) - [PR #26675](https://github.com/BerriAI/litellm/pull/26675)
- **[Bedrock](../../docs/providers/bedrock)**
    - `GET /v1/batches/{batch_id}` forwards `model` from the encoded id (was returning `LiteLLM doesn't support bedrock for 'create_batch'`) - [PR #26814](https://github.com/BerriAI/litellm/pull/26814)
    - Pass-through stream interruption now flushes spend tracking — `GeneratorExit` from client disconnect was dropping per-chunk usage values - [PR #26719](https://github.com/BerriAI/litellm/pull/26719)
    - Replace deprecated Claude 3.7 Sonnet test references with `claude-sonnet-4-5-20250929-v1:0` across 16 test files - [PR #26721](https://github.com/BerriAI/litellm/pull/26721)
- **[Router custom pricing](../../docs/proxy/custom_pricing)**
    - Propagate custom `cost_per_token` from DB `model_info` through the fallback path - [PR #25888](https://github.com/BerriAI/litellm/pull/25888)

---

## LLM API Endpoints

#### Features

- **Workflows API (new)**
    - Durable agent workflow run tracking. New schema (`LiteLLM_WorkflowRun`, `LiteLLM_WorkflowEvent`, `LiteLLM_WorkflowMessage`) and 8 endpoints under `/v1/workflows/runs/...` (create, list, get, patch, append/list events, append/list messages). `session_id` joins to `LiteLLM_SpendLogs.session_id` for free cost attribution. - [PR #26793](https://github.com/BerriAI/litellm/pull/26793)
- **[Vector Stores](../../docs/vector_stores)**
    - Bedrock `retrievalConfiguration` passthrough via `extra_body`, with explicit allow-listing per provider - [PR #26685](https://github.com/BerriAI/litellm/pull/26685)

#### Bugs

- **[Responses API](../../docs/response_api)**
    - `DELETE /openai/responses/{id}` no longer sends `json={}` — Azure now rejects the empty `{}` body with `unexpected_body` - [PR #26949](https://github.com/BerriAI/litellm/pull/26949)
- **Pass-through endpoints**
    - Invoke post-call guardrails on non-streaming pass-through responses (`/vertex_ai/*`, `/openai/*`, `/bedrock/*`); opt-in only when guardrails are configured for the route - [PR #26262](https://github.com/BerriAI/litellm/pull/26262)
    - Inherit caller identity from `litellm_params` metadata when fabricating `UserAPIKeyAuth` for managed-files passthrough batch creation (Anthropic + Vertex AI) - [PR #26831](https://github.com/BerriAI/litellm/pull/26831)
- **Embedding cache**
    - Preserve `prompt_tokens_details` (incl. `image_count`) through the cache round-trip; aggregate per-item details on retrieval; merge in `combine_usage()` for partial cache hits - [PR #26653](https://github.com/BerriAI/litellm/pull/26653)
- **Streaming logging**
    - Backfill streaming hidden response cost into the success log path - [PR #26606](https://github.com/BerriAI/litellm/pull/26606)
- **Cost calculation**
    - Unify `success_handler` typed and dict branches so spend rows stop logging `0` and the budget-overrun reports it caused - [PR #26629](https://github.com/BerriAI/litellm/pull/26629)

---

## Management Endpoints / UI

#### Features

- **Teams**
    - Team-level search-tool credentials: new `search_tools` array on `LiteLLM_ObjectPermissionTable`; per-key permissions validated as a subset of the owning team's; UI selector under team management - [PR #26691](https://github.com/BerriAI/litellm/pull/26691)
- **[Routing Groups](../../docs/proxy/ui/routing_groups)**
    - New **General Settings → Routing Groups** page: create, edit, and delete per-model routing strategies from the dashboard without editing `proxy_config.yaml`. UI-managed groups are persisted and override values defined in YAML; per-group state is rebuilt on save - [PR #27131](https://github.com/BerriAI/litellm/pull/27131)
- **Model Health**
    - Pagination controls on the model health status page - [PR #26826](https://github.com/BerriAI/litellm/pull/26826)
- **CLI / Workers**
    - `--timeout_worker_healthcheck` CLI flag (env `TIMEOUT_WORKER_HEALTHCHECK`) — forwards to uvicorn 0.37.0+ Config kwarg; older uvicorn = warning + no-op; gunicorn / hypercorn paths untouched - [PR #26622](https://github.com/BerriAI/litellm/pull/26622)
- **Memory / lazy loading**
    - Lazy-load optional feature routers on first request (~700 MB lower memory on a two-worker Docker deployment) - [PR #26534](https://github.com/BerriAI/litellm/pull/26534)
    - Lazy-loaded openapi.json front page; spec generation moved to CI with a runtime stub fallback - [PR #26802](https://github.com/BerriAI/litellm/pull/26802)
- **Background jobs**
    - Cleanup job for expired LiteLLM dashboard session keys - [PR #26460](https://github.com/BerriAI/litellm/pull/26460)
- **MCP OAuth**
    - Azure Entra discovery endpoint support - [PR #26584](https://github.com/BerriAI/litellm/pull/26584)

#### Bugs

- **MCP UI**
    - Tool Configuration panel on the MCP server edit page switched from `POST /mcp-rest/test/tools/list` (temp-session preview, requires inline creds) to `GET /mcp-rest/tools/list?server_id=...` (stored credentials). Saved servers with `auth_type` of `api_key` / `bearer_token` / `basic` / `authorization` now load tools without "Unable to load tools — Failed to connect to MCP server." - [PR #26002](https://github.com/BerriAI/litellm/pull/26002)
- **Teams**
    - Per-member rows with `max_budget=NULL` now fall through to team-level enforcement instead of silently disabling it - [PR #26809](https://github.com/BerriAI/litellm/pull/26809)
- **Spend logs**
    - Strip request data from spend-log error messages - [PR #26662](https://github.com/BerriAI/litellm/pull/26662)
- **Vertex retrieve mocked tests**
    - `is_redirect=False` set on mocked retrieve responses - [PR #26844](https://github.com/BerriAI/litellm/pull/26844)

---

## AI Integrations

### Logging

- **General**
    - Opt-in retry settings for the Generic API logger batch send — transient `litellm.Timeout` / `httpx.ConnectTimeout` failures retry instead of dropping the batch - [PR #26645](https://github.com/BerriAI/litellm/pull/26645)
    - Cache GCP IAM token used for Redis (was being regenerated per-connection; synchronous `google-auth` + `google-cloud-iam` calls were freezing the asyncio event loop, causing ~25 s `INCRBYFLOAT` Redis spans in production) - [PR #26441](https://github.com/BerriAI/litellm/pull/26441)
    - Backfill streaming hidden response cost - [PR #26606](https://github.com/BerriAI/litellm/pull/26606)

### Guardrails

- **CyCraft XecGuard (new)**
    - First-class partner guardrail. Multi-policy prompt/response scanning (prompt injection, harmful content, PII, system-prompt enforcement, bias, skills protection) plus RAG context-grounding via `/grounding` - [PR #26011](https://github.com/BerriAI/litellm/pull/26011)
- **Noma v2**
    - `_build_scan_payload` no longer crashes during `post_call` / `during_call` / `during_mcp_call` on `deepcopy(request_data)` failures with unserializable objects (e.g. `uvloop.Loop`) - [PR #26605](https://github.com/BerriAI/litellm/pull/26605)
- **Pass-through**
    - Post-call guardrails on non-streaming pass-through responses (see LLM API Endpoints) - [PR #26262](https://github.com/BerriAI/litellm/pull/26262)

---

## Spend Tracking, Budgets and Rate Limiting

- **Multi-pod budget enforcement**
    - `RedisCache.async_increment` gains `refresh_ttl` opt-in (used by spend counters); `get_current_spend` and `SpendCounterReseed.coalesced` skip stale per-pod in-memory on a clean Redis miss; `ResetBudgetJob` invalidates the Redis counter alongside every DB row reset (keys, users, teams, team members, budgets-linked keys) - [PR #26829](https://github.com/BerriAI/litellm/pull/26829)
- **Cost calc unification**
    - `success_handler` typed + dict branches now compute cost the same way - [PR #26629](https://github.com/BerriAI/litellm/pull/26629)
- **Per-member null budget**
    - Per-member rows with `max_budget=NULL` fall through to team enforcement - [PR #26809](https://github.com/BerriAI/litellm/pull/26809)
- **Bedrock 1-hour cache write pricing**
    - Claude 4.5 / 4.6 / 4.7 Global + US entries gain `cache_creation_input_token_cost_above_1hr` (was undercounting ~60%) - [PR #26800](https://github.com/BerriAI/litellm/pull/26800)
- **`gpt-5.5-pro` corrected pricing**
    - Was double-priced - [PR #26651](https://github.com/BerriAI/litellm/pull/26651)
- **Bedrock pass-through stream interruption**
    - Spend tracking now flushes when client disconnects mid-stream - [PR #26719](https://github.com/BerriAI/litellm/pull/26719)

---

## MCP Gateway

- **Tool prefix**
    - Opt-in `LITELLM_USE_SHORT_MCP_TOOL_PREFIX` env var: switches per-tool prefix from the human-readable server name (`github_onprem-get_repo`) to a deterministic 3-char base62 id derived from `server_id` (`Xy7-get_repo`). Lets long server names stay under the 60-char tool-name limit some model APIs enforce - [PR #26733](https://github.com/BerriAI/litellm/pull/26733)
- **OAuth**
    - Azure Entra discovery endpoint support - [PR #26584](https://github.com/BerriAI/litellm/pull/26584)
    - See **Important Behavior Changes** for public-route detection, OAuth root endpoint visibility, OAuth metadata SSRF guard, and user-scoped credential encryption.

---

## Performance / Loadbalancing / Reliability improvements

- **[Routing Groups (per-model strategies)](../../docs/routing#routing-groups---per-model-strategies)**
    - New `router_settings.routing_groups` schema binds a list of `model_name`s to its own `routing_strategy` and optional `routing_strategy_args`; ungrouped models fall back to the top-level `routing_strategy` (the implicit `default` group, name reserved). Each `model_name` may belong to at most one group — overlap raises `ValueError` at init. Updatable at runtime via `Router.update_settings(routing_groups=[...])` or `/config/update`; per-group state is rebuilt on update - [PR #27022](https://github.com/BerriAI/litellm/pull/27022)
- **Database reconnect**
    - Prisma reconnect no longer blocks the asyncio event loop. Replaces `await self.db.disconnect()` (which calls `subprocess.Popen.wait()` synchronously and freezes the loop for 30–120 s+ in production, failing K8s liveness probes) with SIGTERM → 0.5 s sleep → SIGKILL → fresh `Prisma()` + `connect()`. Direct-reconnect path delegates to `recreate_prisma_client` - [PR #26225](https://github.com/BerriAI/litellm/pull/26225)
    - `call_with_db_reconnect_retry` helper centralizes the reconnect-and-retry-once pattern. Restores the self-heal that 1.83.x lost on `PrismaClient.get_generic_data` (issue [#25143](https://github.com/BerriAI/litellm/issues/25143)) and harden the reconnect state machine - [PR #26756](https://github.com/BerriAI/litellm/pull/26756)
- **Redis IAM token caching**
    - GCP IAM token is no longer regenerated on every Redis connection; a single Redis `INCRBYFLOAT` was taking 25.6 s on a 28.4 s trace in production - [PR #26441](https://github.com/BerriAI/litellm/pull/26441)
- **Config caching**
    - DualCache config parameter reads are cached and batched. End-to-end on Docker, read load drops from 2.8 q/s to 0.7 q/s; improvement scales with pod count. Note: config edits will take longer to propagate (until the cache is invalidated) - [PR #26469](https://github.com/BerriAI/litellm/pull/26469)
- **Memory footprint**
    - Lazy-loaded feature routers - [PR #26534](https://github.com/BerriAI/litellm/pull/26534)
    - Lazy-loaded front page + openapi.json move-to-CI - [PR #26802](https://github.com/BerriAI/litellm/pull/26802)
- **Connection layer**
    - Optional TCP `SO_KEEPALIVE` support on aiohttp's `TCPConnector` - [PR #26730](https://github.com/BerriAI/litellm/pull/26730)
- **CLI**
    - `--timeout_worker_healthcheck` flag for uvicorn worker triage (see Management Endpoints) - [PR #26622](https://github.com/BerriAI/litellm/pull/26622)
- **Test stability**
    - Scope `test_model_alias_map` ERROR-log assertion to LiteLLM logger so `asyncio` records (e.g. `Unclosed client session`) stop flunking the assertion intermittently - [PR #26741](https://github.com/BerriAI/litellm/pull/26741)
    - Replace lazy-load subprocess startup-import diff with static source scan (~13 s instead of timing out past two minutes) - [PR #26934](https://github.com/BerriAI/litellm/pull/26934)
    - Opt model-access E2E tests into `allow_client_mock_response: true` after the request-control hardening - [PR #26941](https://github.com/BerriAI/litellm/pull/26941)
- **Validation**
    - Validate AWS region name on credential intake - [PR #26906](https://github.com/BerriAI/litellm/pull/26906)
    - Drop unsupported `dbName` and `partitionNames` from `MILVUS_OPTIONAL_PARAMS` - [PR #26910](https://github.com/BerriAI/litellm/pull/26910)

---

## General Proxy Improvements

- **CI / Tooling**
    - Support CircleCI "Rerun failed tests" for `local_testing_part1` / `local_testing_part2` / `litellm_router_testing` jobs (was collecting 0 items + exit 123) - [PR #26461](https://github.com/BerriAI/litellm/pull/26461)
    - Correct `min-release-age` value in `.npmrc` files: drop the `d` suffix to keep `npm install` from crashing on npm 11.x with `RangeError: Invalid time value` - [PR #26850](https://github.com/BerriAI/litellm/pull/26850)
- **Pull request template**
    - Add Linear ticket field for internal contributors - [PR #26655](https://github.com/BerriAI/litellm/pull/26655)

---

## New Contributors

- @xinrui-z made their first contribution in [#24294](https://github.com/BerriAI/litellm/pull/24294)
- @Jerry-SDE made their first contribution in [#25249](https://github.com/BerriAI/litellm/pull/25249)
- @Zerohertz made their first contribution in [#25888](https://github.com/BerriAI/litellm/pull/25888)
- @clyang made their first contribution in [#26011](https://github.com/BerriAI/litellm/pull/26011)
- @mverrilli made their first contribution in [#26122](https://github.com/BerriAI/litellm/pull/26122)
- @tuhinspatra made their first contribution in [#26262](https://github.com/BerriAI/litellm/pull/26262)
- @omriShukrun08 made their first contribution in [#26605](https://github.com/BerriAI/litellm/pull/26605)
- @lmcdonald-godaddy made their first contribution in [#26651](https://github.com/BerriAI/litellm/pull/26651)
- @minznerjosh made their first contribution in [#26710](https://github.com/BerriAI/litellm/pull/26710)
- @yassinkortam made their first contribution in [#26730](https://github.com/BerriAI/litellm/pull/26730)
- @sruthi-sixt-26 made their first contribution in [#26814](https://github.com/BerriAI/litellm/pull/26814)

**Full Changelog**: https://github.com/BerriAI/litellm/compare/v1.83.14-stable...v1.84.0

---

## 05/05/2026 (`v1.84.0-rc.1`)

* New Models / Updated Models: 19
* LLM API Endpoints: 6
* Management Endpoints / UI: 22
* AI Integrations (Logging / Guardrails): 3
* Spend Tracking, Budgets and Rate Limiting: 5
* MCP Gateway: 6
* Performance / Loadbalancing / Reliability improvements: 14
* General Proxy Improvements: 2
* Documentation Updates: 1

Subtotal: 78 PRs

## 05/14/2026 (`v1.84.0` — delta on top of rc.1)

* Hardening: 6
* Budget reservation: 2
* Health probes: 3
* MCP: 2
* Routing under `SERVER_ROOT_PATH`: 1
* Tagging & metrics: 2
* Packaging: 2

Subtotal: 18 PRs

Total: 96 PRs
