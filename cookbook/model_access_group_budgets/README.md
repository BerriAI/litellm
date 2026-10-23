# Track a shared monthly model budget

This guide configures a USD 300 calendar-month soft budget on existing Model Access Groups. It requires PostgreSQL and a proxy with the deployment attribution option. It does not filter by provider or grant model access.

## Configure the deployments

Add the same group and `access_group_spend_attribution: deployment` to each deployment whose spend should count. Include every deployment of a model, including fallback deployments. Keep your existing provider parameters.

```yaml
model_list:
  - model_name: budgeted-chat
    litellm_params:
      model: openai/gpt-6.1-sol
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      access_groups: [shared-pool]
      access_group_spend_attribution: deployment

litellm_settings:
  timezone: Europe/Madrid
  budget_reset_time: "00:00"

general_settings:
  database_url: os.environ/DATABASE_URL
  alerting: [webhook]
  alert_types: [budget_alerts]
```

Set `WEBHOOK_URL` to your alert receiver and reload the configuration using your usual deployment process. The timezone and reset time are proxy-wide settings, so check their effect on other budgets first.

Set this option on the models selected by a smart router. A router's classifier or embedding call counts only if its own deployment belongs to the group and enables deployment attribution. Setting the option only on the outer router is insufficient.

Leave the option unset, or set it to `permission`, to retain the existing behavior. In that mode, only groups explicitly granted by an applicable caller allowlist are charged. Deployment attribution changes accounting, not authorization or hard-budget enforcement. Existing permission-based hard limits still apply.

## Create the budget

Use an administrative key against your own proxy.

```bash
curl --fail-with-body -X PUT "$LITELLM_URL/access_group/shared-pool/budget" \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"soft_budget":300,"budget_duration":"1mo"}'
```

For a new pool, the response has `max_budget: null` and a `budget_reset_at` on the first of the next month in the configured timezone. A soft budget sends alerts without rejecting requests. Other key, team, organization or deployment limits remain effective. If reusing a pool that already has a hard budget, remove its old budget first or explicitly clear that limit through the budget API. Do not assume setting a soft limit clears an existing hard limit.

The group can come from YAML, PostgreSQL, or both. PostgreSQL remains required for budgets and recorded spend. This does not move YAML models into PostgreSQL or rewrite the YAML file.

## Verify the result

```bash
curl --fail-with-body "$LITELLM_URL/access_group/list" \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY"
curl --fail-with-body "$LITELLM_URL/access_group/shared-pool/budget" \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY"
```

Call a configured model using an unrestricted or wildcard key, then read the budget again after the spend queue flushes. Confirm that the increment matches the priced request. Multiple applicable groups each receive the full cost, but duplicate membership in one group is counted once. There is no extra charge for a permission grant to the same group.

Crossing the soft limit triggers the existing budget alert delivery mechanism after the cost callback updates the counters. Configure `budget_alerts` and a supported destination. Delivery uses the existing alert TTL and is best effort. Concurrent workers can send duplicate alerts, and delivery is not an exactly-once notification service.

## Changes and operational limits

Budget rows and recorded spend survive proxy restarts. Removing the last YAML member does not erase the budget. The group remains listed with no deployments, and its budget can still be read or deleted. Manage YAML membership in YAML. The group membership mutation endpoints still manage PostgreSQL deployments and reject replacement or deletion of groups containing YAML members. Budget deletion remains available.

Attribution resolves the actual deployment ID against the serving router at cost-callback time. The resolved groups are shared by the counter and queued database update. A later YAML reload cannot change that queued charge. A reload *before* the callback can change membership, and removing a deployment before its callback can lose deployment-only attribution. Drain in-flight requests before removing or changing those deployments. Keep configuration consistent across replicas and spend workers. This feature does not retroactively reclassify historical spend.

Only costs reported by the existing LiteLLM cost pipeline can be counted. Streaming uses its final or recovered usage event. Image generation needs supported pricing. Asynchronous video, batches, background responses and provider pass-through require their existing completion-cost tracking and a resolvable deployment ID. Status polls without a charge must not be treated as new inference. This example does not establish coverage for every provider, endpoint or plugin. Inspect spend logs and validate the specific route before relying on its totals.

The existing spend queue, atomic database increments, event deduplication and reset job remain in use. Queue flushes and resets are asynchronous. This is an operational budget, not an invoice ledger with event-time month boundaries, guaranteed delivery during crashes, or retroactive recovery of missing provider costs. Use the existing spend-worker durability controls and reconcile against provider billing where needed.
