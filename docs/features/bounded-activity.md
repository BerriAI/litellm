# Bounded Activity summaries

Entity Activity tabs use a bounded summary of existing daily activity records. The summary preserves stored daily and range totals while limiting the model breakdown returned to the browser.

Use `bounded=true` on `GET /user/daily/activity/aggregated` or `GET /team/daily/activity/aggregated`. Existing callers that omit this parameter keep the original response and behavior. Existing user, team, API-key and model filters and permissions still apply; ordinary team members retain their existing self-only view.

Provide `start_date` and `end_date` in `YYYY-MM-DD` format. The inclusive range may contain at most 366 days. The response contains:

- `results`: exact metrics for each recorded day, without per-day dimension dictionaries.
- `metadata`: exact totals across the full matching range.
- `model_breakdown`: null unless `include_model_breakdown=true`.

With a model breakdown, `model_limit` selects the highest-spend models across the full range. It defaults to 20 and accepts 1–100. `total_models` counts all matching model identities. The separate `other` object contains the combined metrics and count of every remaining model. It is null when no models remain and is never used as a model identity. A real model named `Other` remains a distinct model. Identifiers longer than 1,024 characters are included in the `other` aggregate, preserving their totals without truncating an identity or returning an oversized name.

For example, request a key's bounded summary with:

```text
GET /user/daily/activity/aggregated?bounded=true&include_model_breakdown=true&model_limit=20&api_key=example-reporting-key-id&start_date=2026-11-01&end_date=2026-11-04
```

Use a reporting key ID, not a secret API key. The model contributions plus `other` sum to the complete range totals. Derived PTU-hours and guardrail or model-group breakdowns are not part of this summary; use the existing Usage view for those breakdowns.

The SQL query limits result rows before the database driver materializes them. At the maximum range and model limit it returns at most 468 rows; without models it returns at most 367. Queries also have bounded concurrent admission, statement time, lock wait and working memory. A busy or over-budget request returns `503` with `Retry-After: 5`. Retry later or narrow the date range. These limits bound the summary response and query resources; query cost still depends on the number of matching stored activity rows.
