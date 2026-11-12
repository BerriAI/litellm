# ADR 002: Task Insights reads applied ledger deliveries into its own observation table

Status: Accepted

## Context

Task Insights reports volume, cost and latency per task and model deployment. The source of truth for what a request cost is `LiteLLM_SpendLogDelivery`, the ledger the accounting flush applies to spend totals. Its `accounting_payload` carries the charge, the billed model, the team and the timing, but no prompt text and no deployment id, and the flush deletes applied non-batch deliveries after `LITELLM_ACCOUNTING_DELIVERY_RETENTION_DAYS`. The prompt text and deployment id live in `LiteLLM_SpendLogs`, keyed by request id, which holds one row per request even when several deliveries billed it

## Decision

When `LITELLM_TASK_INSIGHTS_ENABLED` is set, the accounting flush copies every applied delivery into `LiteLLM_TaskObservation`, keyed by delivery id, after the billing transaction commits and under the same pod lock as cleanup. The copy joins the request log for the deployment id, session and bounded excerpts of the latest and first user text, and ties a delivery to the deployment only when the delivery's billed model is the model the request log names. Cleanup then deletes only deliveries that have an observation. A failed copy is logged and retried on the next flush, it never touches spend. With the flag unset, the flush and cleanup run exactly as before

## Alternatives Considered

**Write the observation inside the billing transaction**. A failure in analytics would roll back the charge

**Read `LiteLLM_SpendLogs` directly**. It has no per-delivery charge, so retries and failed attempts that were billed would be lost or double counted, and it can be disabled per request while the ledger still bills

**Extend `accounting_payload` with the prompt text**. It would put prompt content into the billing ledger and grow every delivery row

## Consequences

Benefits:

- Analytics count what was billed, per delivery, and survive ledger retention
- The billing path and its tests are unchanged when the feature is off

Costs:

- When the feature is on and the copy keeps failing, applied deliveries accumulate, which the existing delivery storage alert reports
- Observations hold bounded request excerpts and have no retention of their own yet
- A delivery billed on a model other than the one the request log names keeps an unknown deployment
