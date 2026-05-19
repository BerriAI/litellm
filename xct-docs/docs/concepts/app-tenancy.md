---
sidebar_position: 6
---

# App-Tenancy

An **XCT App** is a first-class consumer identity — xct-chat, xct-home,
xct-agent-desktop, etc. Each app has:

- an `app_id` (UUID PK)
- an OAuth `client_id` (public) + `client_secret` (hashed, returned once)
- a redirect URI whitelist (exact-match enforced)
- a default team + default scopes
- an optional `capability_scope_id` pointing at an access group
- per-app `rpm_limit` and `daily_budget`

## Why apps and not just keys/teams

A team is "who pays"; an app is "what's calling". A single user can be in
xct-chat and xct-home, but those two apps should see different capabilities.
Per-app metrics also need a stable dimension that doesn't change when a
user rotates their personal key.

## Provisioning

Admin only:

```http
POST /v1/xct-apps
{
  "app_name": "xct-chat",
  "display_name": "XCT Chat",
  "redirect_uris": ["https://chat.xct.test/oauth/callback"],
  "default_team_id": "t-XYZ",
  "default_scopes": ["read", "write"],
  "capability_scope_id": "grp-chat-allowed"
}
→ 200 { ..., client_secret: "<returned ONCE>" }
```

## OAuth handshake (PKCE)

```
Browser → GET /oauth/authorize?client_id=xct_abc&redirect_uri=…
                              &response_type=code&state=xyz
                              &code_challenge=…&code_challenge_method=S256
       → 302 redirect_uri?code=…&state=xyz
App backend → POST /oauth/token grant_type=authorization_code
                           code=…&code_verifier=…&redirect_uri=…
       → { access_token, refresh_token, expires_in: 3600 }
```

Browser flows: **PKCE required** (no client_secret).
Server-to-server flows: client_secret accepted and validated.

## How `app_id` propagates

```
LiteLLM_VerificationToken row.app_id
    ▶ UserAPIKeyAuth.app_id
        ▶ metadata.user_api_key_app_id (stamped in chat_completion)
            ▶ LiteLLM_SpendLogs.app_id (S6-01)
            ▶ litellm_capability_*_total{app_id} (S6-02)
            ▶ capability.invoked webhook (S6-06)
            ▶ /v1/capabilities response (narrowed by capability_scope_id, S4-08)
```

### Header fallback

`x-xct-app-id` header sets `app_id` ONLY when the token row doesn't already
carry one. Token-baked `app_id` always wins — otherwise a leaked admin
key could impersonate any app.

## Capability scope intersection

When a token has `app_id`, the `/v1/capabilities` response is further
narrowed by `app.capability_scope_id` (an `LiteLLM_AccessGroupTable` row
with `access_model_names` / `access_agent_ids` / `access_mcp_server_ids`).

Visible = caller's permission set **∩** app's capability scope.

Empty scope or no `capability_scope_id` = no narrowing.

## Revoking & rotating

```http
POST /v1/xct-apps/{app_id}/rotate-secret    →  new cleartext, old immediately dead
POST /oauth/revoke {token: "..."}            →  RFC 7009; always 200 (no enumeration)
POST /oauth/introspect {token, client_id, client_secret}  →  current status
DELETE /v1/xct-apps/{app_id}                 →  hard-delete; tokens stay for audit
```

## See also

- **[Quickstart: xct-chat OAuth](../quickstart/xct-chat.md)**
- **[Recipe: OAuth PKCE in React](../recipes/oauth-pkce-react.md)**
