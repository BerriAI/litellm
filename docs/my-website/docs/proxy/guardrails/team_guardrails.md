# Team-based guardrails

:::info

This is an Enterprise feature.
[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/cx9p-5yf-2nm/litellm-introductions)

:::

Team admins can create guardrails scoped to their team. Those guardrails are only available to that team (and to proxy admins). This mirrors [team model onboarding](/proxy/team_model_add).

## Create a team guardrail

Use the same `/guardrails` POST endpoint with a team API key and include `team_id` in the body (top-level or inside `guardrail` / `guardrail_info`):

```bash
curl -X POST "http://localhost:4000/guardrails" \
  -H "Authorization: Bearer <team_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "guardrail": {
      "guardrail_name": "my-team-content-filter",
      "litellm_params": {
        "guardrail": "litellm_content_filter",
        "mode": "pre_call",
        "default_on": true
      },
      "guardrail_info": { "description": "Team content filter" }
    },
    "team_id": "e59e2671-a064-436a-a0fa-16ae96e5a0a1"
  }'
```

- **Proxy admin**: Can create global guardrails (omit `team_id`) or team-scoped guardrails (set `team_id`).
- **Team admin**: Must set `team_id` to a team they administer; the guardrail is then only available for that team.

## List guardrails by team

`GET /v2/guardrails/list` supports optional query params:

- **`team_id`** – When used with `view=current_team`, returns global guardrails plus guardrails for that team.
- **`view`** – `all` (default): all guardrails; `current_team`: global + guardrails for the given `team_id`.

```bash
# All guardrails (admin)
curl -X GET "http://localhost:4000/v2/guardrails/list" -H "Authorization: Bearer <admin_key>"

# Global + team guardrails for a team
curl -X GET "http://localhost:4000/v2/guardrails/list?team_id=e59e2671-a064-436a-a0fa-16ae96e5a0a1&view=current_team" \
  -H "Authorization: Bearer <team_api_key>"
```

Each guardrail in the response includes `team_id` (null for global) so the UI can show scope.

## Request-time behavior

For a request with a team API key, guardrails are resolved by name with team precedence:

1. If a guardrail with that name exists for the request’s team, it is used.
2. Otherwise the global guardrail with that name (if any) is used.

So team-scoped guardrails override global ones for that team; they are only available to that team.
