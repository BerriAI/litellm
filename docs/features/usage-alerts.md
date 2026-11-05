# Daily usage alerts

Daily usage alerts compare a completed UTC day with earlier recorded days. They cover spend, request count, average latency, and failed-request count for the gateway, a key, a user, a team, or a model. Open an incident to inspect its saved comparison and follow its Activity link to the corresponding usage view.

This feature requires a database. Apply the normal LiteLLM database migrations when deploying a version that includes usage alerts. The API and dashboard use the same persisted rules and incidents; no enterprise license is required.

## Read incidents

Set `LITELLM_API_BASE` to your gateway's base URL and `LITELLM_MANAGEMENT_KEY` to a management-capable credential. Keep the key in your environment.

```bash
curl "$LITELLM_API_BASE/usage/alerts?status=open&limit=50" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
```

The response has `data`, `total`, `limit`, and `offset`. Each incident includes its scope, metric, direction, status, UTC day, observed value, baseline value, threshold, baseline date range, and `activity_path`. The Activity path is relative to the dashboard mount: for example, `/api-keys?...` is opened below the dashboard's `/ui` mount.

`GET /usage/alerts/{alert_id}` returns one incident. List filters are `scope_type`, `scope_id`, `metric`, `status`, `start_date`, and `end_date`. Dates use `YYYY-MM-DD`; date filters include both endpoints. The maximum page size is 100.

## Create a rule

This example alerts when a team's daily spend is at least three times its baseline and the absolute change is at least $10. Replace `team-example` with the team's ID.

```bash
curl "$LITELLM_API_BASE/usage/alert-rules" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Team daily spend",
    "scope_type": "team",
    "scope_id": "team-example",
    "metric": "spend",
    "direction": "increase",
    "multiplier": 3,
    "baseline_days": 7,
    "min_history_days": 3,
    "min_requests": 100,
    "min_value": 10,
    "enabled": true
  }'
```

The baseline is the mean across recorded days in the preceding window. Missing reference days are unknown. At least `min_history_days` recorded days are required. A decrease uses `baseline / multiplier`; a multiplier of 2 detects a value at or below half the baseline. `min_value` is the minimum absolute difference in the metric's own units.

| Metric | Unit |
| --- | --- |
| `spend` | USD |
| `requests` | Request count |
| `latency` | Average response time in milliseconds |
| `errors` | Failed-request count |

The evaluator runs against daily aggregates. An hourly scheduler does not make these rolling hourly alerts. A complete UTC day and sufficient ingestion coverage are required before detecting drops; a missing flush must not become a false traffic-drop incident. Rule responses expose `last_evaluated_day` and `evaluation_status` so you can distinguish a completed evaluation from insufficient history or coverage.

Use `scope_type: "deployment"` with no `scope_id` for the gateway. For `model`, use the underlying model identity recorded in Activity. For `key`, use its reporting ID, never the secret API key. Omitting `scope_id` for key, user, team, or model evaluates all entities of that type and requires a gateway administrator.

## Manage rules and incidents

- `GET /usage/alert-rules` lists rules; `GET /usage/alert-rules/{rule_id}` reads one.
- `PUT /usage/alert-rules/{rule_id}` replaces the rule configuration. Send the configuration fields from the create example, including any notification settings you want to retain. Set `enabled` to `false` to pause it.
- `DELETE /usage/alert-rules/{rule_id}` removes a rule from evaluation and preserves existing incidents. Deleted defaults stay deleted.
- `PATCH /usage/alerts/{alert_id}` with `{"status":"dismissed"}` dismisses an incident. `open` and `resolved` are also accepted.

Consecutive breached days update one incident. Re-evaluating the same day does not reopen a dismissed or resolved incident. After recovery, a later breach starts a new incident. Editing a rule does not rewrite historical comparisons.

Rules can specify `notification_channels: ["slack", "email"]` and `email_recipients: ["alerts@example.com"]`. Notification transports must already be configured on the gateway. An empty channel list keeps incidents available through the API and dashboard. Successful delivery is recorded per channel so a failed channel can retry independently.

## Access

Gateway administrators can manage every scope. Admin viewers can read every scope. Internal users can manage their own user and key scopes; team administrators can manage their team's scope and keys. Team members can read their team's incidents. Read-only users cannot change rules or incident state, and inference-only keys cannot access these endpoints.

Non-admin list requests must specify a scope they can access, for example `?scope_type=user&scope_id=user-example`. A `403` indicates that the credential or scope lacks access. A `503` indicates that the gateway has no database connection.
