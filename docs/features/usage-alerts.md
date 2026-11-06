# Alerts

The Alerts page combines model retirements with daily Performance alerts. Open an alert to inspect its saved comparison, supporting signals, and Activity for the affected resource or provider/model cohort. Related signals can share one investigation without discarding their individual records. Grouping describes observed relationships; it does not establish a root cause.

Performance comparisons cover completed UTC days. Retirement issues show recent model usage and the provider's retirement schedule, including today's usage when present.

This feature requires a database. Apply the normal LiteLLM database migrations when deploying a version that includes usage alerts. The API and dashboard use the same persisted rules and incidents; no enterprise license is required.

## Start with the defaults

Daily Performance defaults cover spend, latency, reliability, and traffic across the gateway, keys, users, teams, and models. They use the previous seven days and the following policies. Retirement uses the existing model-deprecation warning window.

| Family | Default comparison | Minimum history and volume |
| --- | --- | --- |
| Spend | At least 3× daily spend and a $10 increase | Three recorded reference days with a positive spend baseline |
| Latency | At least 3× average latency and a 500 ms increase | Three recorded reference days and at least 100 timed requests in both periods |
| Reliability | At least 3× error rate and a two percentage-point increase | Three recorded reference days and at least 500 daily requests in both periods |
| Traffic increase | At least 10× requests and an increase of 1,000 requests | Three recorded reference days with a positive request baseline |
| Traffic decrease | At most one fifth of baseline requests | Seven recorded reference days, each with at least 500 requests |

New resources need history before they can produce a default spike. A zero spend, request, or latency baseline does not establish a spike. A zero error-rate baseline is usable only after sufficient historical traffic has been observed; the percentage-point floor still applies.

Open **Alerts → Settings** to enable or disable a family or adjust its thresholds across the five scopes. Only enabled custom rules managed by a gateway administrator replace matching defaults for their scope, metric, and direction. An administrator-managed wildcard rule replaces that default for every entity. Rules created by scoped non-admins add alerts without suppressing defaults. Deleted defaults stay deleted.

## When checks run

An initial check is scheduled shortly after startup, followed by hourly wakeups. The evaluator checks the latest completed UTC day after a one-hour grace period. A successful aggregate flush after the grace period is required before either direction is finalized. Completed rule versions are skipped on later wakeups. These are daily comparisons, not rolling hourly alerts.

Each run has a shared 60-second work budget, including up to five seconds for model-retirement checks. Performance evaluation reads existing daily aggregates in pages of 10 entities, with a short pause between pages. Proxy replicas share a database lease for this job. It adds no request-time counters or per-request database writes.

Completed pages save a checkpoint. When time or database capacity runs out, unfinished work resumes from its saved position on a later wakeup. A page interrupted before its checkpoint can be repeated safely. This is best-effort progress: a large or busy deployment can need several hourly wakeups, and repeated query timeouts can prevent a page from advancing. The hourly schedule does not guarantee that every resource or notification finishes within an hour.

Completed key alerts can appear while other resources are still being checked or related signals are being grouped. **Monitoring** shows **Catching up** for unfinished work and explains that it resumes automatically. Rule status `delayed` means work remains; `insufficient_history` means a completed comparison lacked usable baseline data, while `insufficient_coverage` means the required usage-data coverage is missing. A partial pass is not reported as a completed scan.

Decrease rules need a successful background aggregate-flush sample for every hour of the target day and another sample after the grace period. Until that coverage exists, their status is `insufficient_coverage`. A key with no requests can then produce a request or spend drop; missing latency samples never become zero latency. Coverage shows that an aggregate writer was healthy each hour, not that every replica delivered every request.

The baseline averages recorded reference days; it does not turn missing reference days into zero. Average latency is weighted by timed request count, using the same successful-request duration totals as Activity. `baseline_sample_days` records how many reference days contributed. Cost, traffic, and latency comparisons require a positive baseline. Reliability can compare against a recorded 0% error rate when historical and current request volume meet the rule's minimum. Missing history and low-volume comparisons remain `insufficient_history`.

## How related signals form one alert

A key with unusually high spend gets a spend issue, including when it makes only a few expensive requests. Its comparison includes request volume so you can distinguish more traffic from more expensive traffic. A traffic-increase signal joins that spend issue when both increases use the same reference window and their ratios are within 20% of each other.

Default user, team, model, and gateway signals can join a key issue when they describe the same key traffic. For spend and request increases, they can also join when that key accounts for at least 80% of the source's positive growth. Each supporting signal retains its own comparison. A reliability or latency change remains a separate issue from increased spend.

Provider latency grouping requires measured deterioration within established key/model traffic: at least two keys per model, two models, and 80% coverage of current timed requests. A change in the mix of fast and slow models alone is insufficient. A source joins only when at least 80% of its timed traffic independently meets the shared pattern; a source with more than twice the provider cohort's latency ratio stays separate. The issue keeps its original provider/model cohort until recovery. These relationships are evidence for a shared investigation, not proof of a provider outage or root cause. Custom rules retain their configured resource scope.

## Read and investigate alerts

Set `LITELLM_API_BASE` to your gateway's base URL and `LITELLM_MANAGEMENT_KEY` to a management-capable credential. Keep the credential in your environment.

```bash
curl "$LITELLM_API_BASE/issues?status=open&limit=50" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
```

The response contains `data`, `total`, `limit`, and `offset`. Pagination and filtering happen on the combined inbox. Filters are `kind`, `family`, `status`, `scope_type`, and `scope_id`. Families are `retirement`, `latency`, `reliability`, `spend`, and `traffic`; kinds are `model_deprecation` and `performance`. An offset beyond the last page still returns the correct total. There is no default history cutoff on open issues.

Each issue has an ID, status, readable title, current resource label, typed `cohort`, date range, grouping reason, and saved comparison values. The cohort is one resource, one provider, or exact provider/model pairs. A provider/model pair is an intersection, not a list of independently interchangeable filters. Resource names can change; IDs remain the identity.

Use the ID from the list to inspect an issue:

```bash
ISSUE_ID="issue-example"
curl "$LITELLM_API_BASE/issues/$ISSUE_ID" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
curl "$LITELLM_API_BASE/issues/$ISSUE_ID/activity" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
curl "$LITELLM_API_BASE/issues/$ISSUE_ID/signals?limit=50" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
```

Replace `issue-example` with an ID returned by `/issues`. Activity uses the issue's saved date range by default. Optional `start_date` and `end_date` are inclusive UTC dates and can span at most 366 days. `/keys` and `/users` provide paginated Activity rows with `limit` and `offset`. Their latency totals are `total_response_time_ms` and `timed_requests`; divide the former by the latter only when timed requests are present.

Alert Activity returns exact daily totals in `results` and exact range totals in `metadata`. It does not return the general Usage API's per-day dimension dictionaries. Key issues also return `model_breakdown`, with up to 20 models by range spend by default. Set `model_limit` between 1 and 100 to change that limit. `total_models` counts every matching model; `other` contains the combined metrics and `model_count` of all remaining models, or null when none remain. It is an aggregate, not a model identifier. Adding the displayed models and `other` recovers the full totals. Model identifiers longer than 1,024 characters are included in `other` without truncation. Non-key issues return null for `model_breakdown`. Activity queries use a bounded execution budget. A busy or over-budget query returns `503` with `Retry-After: 5`; retry later or choose a shorter date range.

`/impact` reports distinct resource counts and requests within that same Activity range. These counts mean that resources used the cohort, not that each resource suffered a regression. Saved `impact.measured_keys` and `impact.measured_models`, when present, describe cohorts that met the detector's comparison criteria. Resource pages fetch labels only for the selected page, using the latest deleted-key label when needed. They do not load arbitrary key or user metadata. Display aliases are limited to 256 characters; email labels longer than 320 characters are omitted. Resource pages omit IDs longer than 1,024 characters or 4,096 bytes instead of truncating them. Key IDs that need a `raw:` escape prefix reserve four characters within that limit. Oversized linked user or team IDs are omitted from resource metadata. Activity totals still include traffic from omitted resources. Historical team aggregates do not retain user membership, so a team issue has no `/users` view and its user count is null. User issues similarly have no historical team count. Missing attribution remains unknown.

Supporting signals have their own saved comparisons and relationship explanations. Their baseline windows can differ, so do not derive a combined error rate or unit cost by adding their displayed values. Administrators and admin viewers can inspect broader supporting signals. A scoped reader receives only signals for the exact key, user, or team they can access. That restriction applies before the signal count and pagination; broader totals are not disclosed.

A continuing Performance issue retains its initial comparison and cohort while `last_seen` advances. New supporting observations do not rewrite the original evidence. Resource issues recover when their primary metric recovers for that resource; broader supporting observations do not keep them open. Provider issues require a fresh comparison of the saved provider/model cohort. Missing history or coverage does not establish recovery. A later breach starts a new issue. Retirement details refresh as usage and the provider schedule change. A retirement issue with no recent use becomes inactive; actual new traffic can reopen it.

Resolve or dismiss an issue using the current status from the read response:

```bash
curl -X PATCH "$LITELLM_API_BASE/issues/$ISSUE_ID" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"expected_status":"open","status":"resolved"}'
```

A stale status returns `409`; reload before retrying. Performance acknowledgements do not change other issues that share a supporting observation. Retirement acknowledgements use the same state as the existing model-deprecation API. An inactive retirement cannot be manually reopened.

The issue-detail response includes `can_update_status` and `can_view_activity` for the current reader. Clients should show status actions only when `can_update_status` is true and request Activity or resource breakdowns only when `can_view_activity` is true. These fields are null in responses where permission has not been evaluated, such as inbox lists. Every endpoint still checks authorization on each request. Team Activity requires team-admin access or the explicit `/team/daily/activity` permission; ordinary membership can read saved alert evidence but does not grant other members’ activity or key details.

## Configure default families

`GET /issues/families` returns the four Performance families, their enabled counts, common policy values, and `evaluation_status_counts` for enabled rules. A null common value means the underlying scopes have different settings. The top-level `projection_pending` flag indicates that alert grouping has unfinished work, even if no rule is currently `delayed`. This is separate from custom rules and the model-retirement warning window.

This example sets the spend family to a 5× threshold and a $25 minimum absolute increase:

```bash
curl -X PATCH "$LITELLM_API_BASE/issues/families/spend" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled":true,"policies":[{"direction":"increase","multiplier":5,"min_value":25}]}'
```

The update applies atomically to registered, non-deleted built-in rules in that family. Omitted settings remain unchanged. It does not change custom rules or recreate deleted defaults. A concurrent rule edit returns `409`; reload and retry. These controls require a gateway administrator.

## Read resource groups

Set `LITELLM_API_BASE` to your gateway's base URL and `LITELLM_MANAGEMENT_KEY` to a management-capable credential. Keep the key in your environment.

```bash
curl "$LITELLM_API_BASE/usage/alert-groups?status=open&limit=50" \
  -H "Authorization: Bearer $LITELLM_MANAGEMENT_KEY"
```

The response has `data`, `total`, `limit`, and `offset`. `total` counts resource/day groups, including when the requested offset is beyond the last page. Filters are `scope_type`, `scope_id`, `status`, `start_date`, and `end_date`. Status filters apply to individual incidents before grouping, so a group with open and resolved incidents can appear under either filter with the corresponding members. No default date cutoff hides older open incidents.

Each group includes its stable `id`, `scope_type`, exact `scope_id`, current `scope_label` when available, original UTC `day`, `incident_count`, and `activity_path`. Its `metrics` list contains up to five entries with a `metric`, `incident_count`, and the most recently detected matching `representative` incident. These representatives are summaries, not the entire group. Missing or deleted resources keep their original IDs. Names are current display labels, not historical evidence.

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

The baseline is the mean across recorded days in the preceding window. Missing reference days are unknown. At least `min_history_days` recorded days are required. A decrease uses `baseline / multiplier`; a multiplier of 2 detects a value at most half the baseline. `min_value` is the minimum absolute difference in the metric's own units.

| Metric | Unit |
| --- | --- |
| `spend` | USD |
| `requests` | Request count |
| `latency` | Average response time in milliseconds |
| `errors` | Failed-request count |
| `error_rate` | Failed requests divided by requests, as a percentage from 0 to 100 |

Counts come from the existing Activity/spend aggregates, rather than every gateway HTTP outcome. Latency averages successful timed, non-internal requests; it is not time to first token and does not include failed requests. Issue Activity and gateway source incidents use the same aggregates. An error-rate `min_value` of 2 means two percentage points, such as 1% to 3%, not a 2% relative increase.

The evaluator runs against daily aggregates. An hourly scheduler does not make these rolling hourly alerts. A complete UTC day and sufficient ingestion coverage are required before detecting drops; a missing flush must not become a false traffic-drop incident. Rule responses expose `last_evaluated_day` and `evaluation_status` so you can distinguish a completed evaluation from delayed work, insufficient history, or missing coverage.

Use `scope_type: "deployment"` with no `scope_id` for the gateway. For `model`, use the underlying model identity recorded in Activity. For `key`, use its reporting ID, never the secret API key. Omitting `scope_id` for key, user, team, or model evaluates all entities of that type and requires a gateway administrator.

`GET /usage/alert-models?limit=20&search=example` returns paginated model identities recorded in the most recent 30 UTC days. Its `data` entries contain `scope_id` and `label`; `start_date` and `end_date` state the searched window. Pass `next_cursor` as `cursor` to read the next page. A null cursor ends the result. Search matches literal text, not SQL wildcards. This endpoint requires gateway admin or admin-viewer access. Configured aliases and deployment IDs are not reporting-model identities. An existing rule's saved model remains valid even when it falls outside the options window.

## Manage rules and incidents

- `GET /usage/alert-rules` lists rules; `GET /usage/alert-rules/{rule_id}` reads one. Use `kind=custom`, `kind=default`, or `kind=all` to choose which rules to list. Each rule exposes `is_default`. The list's `default_summary` counts total and enabled defaults within the requested scope, independent of `kind`, and reports evaluation-status counts among enabled defaults.
- `PUT /usage/alert-rules/{rule_id}` replaces the rule configuration. Send the configuration fields from the create example, including any notification settings you want to retain. Set `enabled` to `false` to pause it. Built-in rules keep their scope, metric, and direction; create a custom rule to change those fields. Their names, thresholds, notification settings, and enabled state remain editable.
- `DELETE /usage/alert-rules/{rule_id}` removes a rule from evaluation and preserves existing incidents. Deleted defaults stay deleted.
- `PATCH /usage/alerts/{alert_id}` with `{"status":"dismissed"}` dismisses an incident. `open` and `resolved` are also accepted.

Consecutive breached days keep one incident. Its original day, comparison values, and Activity link stay fixed; `last_seen` advances. Re-evaluating the same day does not reopen a dismissed or resolved incident. After recovery, a later breach starts a new incident. Editing a rule does not rewrite historical comparisons.

Rules can specify `notification_channels: ["slack", "email"]` and `email_recipients: ["alerts@example.com"]`. Notification transports must already be configured on the gateway. An empty channel list keeps incidents available through the API and dashboard.

Issue delivery is recorded per channel and destination. A repeated projection does not resend a successful delivery. Failed or unconfigured deliveries become eligible for retry after one hour and are retried on a later background pass with available capacity. Delivery pages contain at most 10 entries, and each transport attempt is limited by the remaining run budget, up to 30 seconds. The worker rechecks the issue, source signal, and current rule immediately before transport. Disabling a rule or acknowledging an issue cannot recall a send already in flight. Delivery is at least once: an ambiguous transport timeout, or partial success across several configured Slack webhooks, can cause a retry after a destination accepted a message.

Configure `PROXY_BASE_URL` for clickable links in email and Slack. Default rules notify through Slack only when `usage_alerts` is explicitly enabled in the gateway's alert types when those rules are first registered. Otherwise they start with in-app/API incidents. To enable notifications after the defaults already exist, edit their channels through the API or dashboard; changing the gateway alert types does not rewrite saved rules. Explicit rule channels use the configured transport without an additional alert-type switch.

If legacy `user_spend_anomalies` notifications are enabled, the default daily user-spend increase rule stays in-app to avoid overlapping notifications. Email requires recipients on the rule.

## Resource limits and delayed work

Alert investigations and resource-page Activity tabs limit returned rows before loading results into the application. For a key, these views show exact totals and a bounded model breakdown: 20 named models by default, at most 100, plus a separate **Other** aggregate for the remaining models. Activity ranges span at most 366 days. A small result page can still require reading many aggregate rows; page size is not a database scan limit.

Guarded queries allow at most two foreground queries and one background query at a time, coordinated across workers sharing the database. They do not build an unbounded waiting queue. Each query has a statement timeout of at most five seconds, a lock wait of at most 100 ms, and a 1 MiB PostgreSQL `work_mem` setting, with stricter existing limits preserved. Background query deadlines also shrink to the remaining run budget. PostgreSQL `work_mem` applies per operation, not to total application or database memory.

When an Activity query is busy or exceeds its budget, the API returns `503` with `Retry-After: 5`. Background work keeps its checkpoint and tries again on a later wakeup. These limits bound work per attempt; database size, query plans, existing gateway memory, and concurrent serving traffic still affect overhead and completion time. Persistent **Catching up** means monitoring has not finished, not that the remaining resources are healthy.

## Access

Gateway administrators can manage every scope. Admin viewers can read every scope. Internal users can manage their own user and key scopes; team administrators can manage their team's scope and keys. Scoped non-admins can create additive rules without notification delivery. Only gateway administrators can configure notification destinations or change operator-managed rules. Team members can read their team's incidents. Team-wide Activity requires team-administrator access or the existing team daily-activity permission. Read-only users cannot change rules or incident state, and inference-only keys cannot access these endpoints. Read-only `info_routes` grants do not authorize mutations.

Non-admin list requests must specify a scope they can access, for example `?scope_type=user&scope_id=user-example`. A `403` indicates that the credential or scope lacks access. A `503` can indicate a missing database connection or unavailable analytics capacity; capacity responses include `Retry-After`.

The dashboard's Alerts page is available to administrators and admin viewers. Other authorized roles can use the scoped API.
