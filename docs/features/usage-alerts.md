# Daily Performance alerts

Performance alerts compare a completed UTC day with earlier recorded days. They cover spend, request count, average latency, and failed-request count for the gateway, a key, a user, a team, or a model. The Performance tab groups alerts for the same resource and original UTC day. Open a resource to inspect its saved comparison and Activity. The API paths remain under `/usage`.

This feature requires a database. Apply the normal LiteLLM database migrations when deploying a version that includes usage alerts. The API and dashboard use the same persisted rules and incidents; no enterprise license is required.

## Read resource groups

Set `LITELLM_API_BASE` to your gateway's base URL and `LITELLM_MANAGEMENT_KEY` to a management-capable credential. Keep the key in your environment.

```bash
curl "$LITELLM_API_BASE/usage/alert-groups?status=open&limit=50" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
```

The response has `data`, `total`, `limit`, and `offset`. `total` counts resource/day groups, including when the requested offset is beyond the last page. Filters are `scope_type`, `scope_id`, `status`, `start_date`, and `end_date`. Status filters apply to individual incidents before grouping, so a group with open and resolved incidents can appear under either filter with the corresponding members. No default date cutoff hides older open incidents.

Each group includes its stable `id`, `scope_type`, exact `scope_id`, current `scope_label` when available, original UTC `day`, `incident_count`, and `activity_path`. Its `metrics` list contains up to four entries with a `metric`, `incident_count`, and the most recently detected matching `representative` incident. These representatives are summaries, not the entire group. Missing or deleted resources keep their original IDs. Names are current display labels, not historical evidence.

To inspect every matching rule, use `GET /usage/alerts` with the group's scope and `start_date` and `end_date` both set to its `day`. Add `metric` and `status` as needed and paginate that response. Rules can have different baseline windows and thresholds, so their comparisons must not be combined into an error rate or a cost-per-request estimate.

The group Activity link uses a representative's saved date range and alert context. The path is relative to the dashboard mount: for example, `/api-keys?...` is opened below the dashboard's `/ui` mount. A continuing episode retains its original day and evidence; the grouped row does not describe today's traffic.

Resolve or dismiss all matching incidents in one resource/day group with an expected current status:

```bash
curl -X PATCH "$LITELLM_API_BASE/usage/alert-groups" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"scope_type":"team","scope_id":"team-example","day":"2026-11-04","expected_status":"open","status":"resolved"}'
```

The response contains `updated_count`. Only incidents in that exact group that still have `expected_status` are updated atomically. Other statuses remain unchanged; repeating the same request returns zero after all matching incidents have changed. Use `scope_type: "deployment"` with `scope_id: null` for the gateway.

## Read individual incidents

`GET /usage/alerts` retains the individual-incident view. Each incident includes its scope, metric, direction, status, UTC day, observed value, baseline value, threshold, baseline date range, and `activity_path`.

`GET /usage/alerts/{alert_id}` returns one incident. List filters are `scope_type`, `scope_id`, `metric`, `status`, `start_date`, and `end_date`. Dates use `YYYY-MM-DD`; date filters include both endpoints. The maximum page size is 100.

## Create a rule

This example alerts when a team's daily spend is more than three times its baseline and the absolute change is at least $10. Replace `team-example` with the team's ID.

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

The baseline is the mean across recorded days in the preceding window. Missing reference days are unknown. At least `min_history_days` recorded days are required. A decrease uses `baseline / multiplier`; a multiplier of 2 detects a value below half the baseline. `min_value` is the minimum absolute difference in the metric's own units.

| Metric | Unit |
| --- | --- |
| `spend` | USD |
| `requests` | Request count |
| `latency` | Average response time in milliseconds |
| `errors` | Failed-request count |

Counts come from the existing Activity/spend aggregates, rather than every gateway HTTP outcome. Latency averages successful timed, non-internal requests; it is not time to first token and does not include failed requests. Gateway incidents link to Key Activity, which uses the same aggregates.

The evaluator runs against daily aggregates. An hourly scheduler does not make these rolling hourly alerts. A complete UTC day and sufficient ingestion coverage are required before detecting drops; a missing flush must not become a false traffic-drop incident. Rule responses expose `last_evaluated_day` and `evaluation_status` so you can distinguish a completed evaluation from insufficient history or coverage.

Use `scope_type: "deployment"` with no `scope_id` for the gateway. For `model`, use the underlying model identity recorded in Activity. For `key`, use its reporting ID, never the secret API key. Omitting `scope_id` for key, user, team, or model evaluates all entities of that type and requires a gateway administrator.

`GET /usage/alert-models?limit=20&search=example` returns paginated model identities recorded in the most recent 30 UTC days. Its `data` entries contain `scope_id` and `label`; `start_date` and `end_date` state the searched window. Pass `next_cursor` as `cursor` to read the next page. A null cursor ends the result. Search matches literal text, not SQL wildcards. This endpoint requires gateway admin or admin-viewer access. Configured aliases and deployment IDs are not reporting-model identities. An existing rule's saved model remains valid even when it falls outside the options window.

## Manage rules and incidents

- `GET /usage/alert-rules` lists rules; `GET /usage/alert-rules/{rule_id}` reads one. Use `kind=custom`, `kind=default`, or `kind=all` to choose which rules to list. Each rule exposes `is_default`. The list's `default_summary` counts total and enabled defaults within the requested scope, independent of `kind`, and reports evaluation-status counts among enabled defaults.
- `PUT /usage/alert-rules/{rule_id}` replaces the rule configuration. Send the configuration fields from the create example, including any notification settings you want to retain. Set `enabled` to `false` to pause it. Built-in rules keep their scope, metric, and direction; create a custom rule to change those fields. Their names, thresholds, notification settings, and enabled state remain editable.
- `DELETE /usage/alert-rules/{rule_id}` removes a rule from evaluation and preserves existing incidents. Deleted defaults stay deleted.
- `PATCH /usage/alerts/{alert_id}` with `{"status":"dismissed"}` dismisses an incident. `open` and `resolved` are also accepted.

Consecutive breached days update one incident. Re-evaluating the same day does not reopen a dismissed or resolved incident. After recovery, a later breach starts a new incident. Editing a rule does not rewrite historical comparisons.

Rules can specify `notification_channels: ["slack", "email"]` and `email_recipients: ["alerts@example.com"]`. Notification transports must already be configured on the gateway. An empty channel list keeps incidents available through the API and dashboard. Successful delivery is recorded per channel so a failed channel can retry independently.

## Access

Gateway administrators can manage every scope. Admin viewers can read every scope. Internal users can manage their own user and key scopes; team administrators can manage their team's scope and keys. Team members can read their team's incidents. Read-only users cannot change rules or incident state, and inference-only keys cannot access these endpoints.

Non-admin list requests must specify a scope they can access, for example `?scope_type=user&scope_id=user-example`. A `403` indicates that the credential or scope lacks access. A `503` indicates that the gateway has no database connection.
