---
sidebar_position: 10
---

# Debug: `/v1/capabilities` returns empty

By design, an unscoped key sees nothing. Walk through these in order.

## 1. Are you admin?

```python
caps = xct.capabilities.list()
print(caps["caller"])     # {"is_admin": True/False, ...}
```

If `is_admin` is `false`, the response is filtered. Use a master key once
to confirm the registry actually has entries — if THAT returns full lists,
the issue is scoping.

## 2. Check the token's grants

Open the key in **Keys** dashboard:

- `models[]` — empty? You won't see any models. Defaults to the team's
  `models[]` if your key has none.
- `object_permission` →
  - `mcp_servers[]` / `mcp_access_groups[]` — empty? No MCPs visible.
  - `agents[]` / `agent_access_groups[]` — empty? Only public + owned agents.

## 3. Check the team

If your team is misconfigured the key inherits nothing.

```http
GET /team/info?team_id=t-XYZ
```

Look at:
- `models[]`
- `object_permission`
- `access_group_ids[]`

## 4. App-tenancy intersect (S4-08)

If your token has `app_id` set:

```python
print(caps["caller"]["app_id"])  # "xct-chat"
```

Then the response was further narrowed by `app.capability_scope_id`. Check
the app:

```http
GET /v1/xct-apps/{app_id}
```

`capability_scope_id` points at an `LiteLLM_AccessGroupTable` row. The
caller's permissions are **intersected** with that group's allow lists.
Empty intersection → empty response.

To temporarily bypass: PATCH the app and set `capability_scope_id: null`.
Or remove `app_id` from the token (use a different key without app
binding).

## 5. The cache

Result is cached for 60s per (token, app_id). If you just changed grants
and don't see them, wait — or punch the entry server-side via
`invalidate_capabilities_cache_for_caller`.

## 6. Public-anonymous variant

`GET /.well-known/xct-capabilities` only returns entities flagged
`is_public`. Empty there usually means no `litellm.public_*_groups`
config has been set.

## 7. The proxy log

`LITELLM_LOG=DEBUG` shows the per-collector failure path:

```
capabilities: model filter fell back to []: ...
```

Every collector swallows exceptions to never break the response — but
they log the cause.
