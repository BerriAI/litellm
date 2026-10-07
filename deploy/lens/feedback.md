# Response usefulness feedback

Lens accepts integer usefulness ratings from **0 through 10 inclusive**, with an optional comment. `0` means not useful; a response with no feedback is **unrated**, not zero. In Lens, open a trace, select a span, and view its recorded output to see the Usefulness section: average, count, distribution, and your editable rating.

## Prerequisites and authorization

- Upgrade the proxy and dashboard together. Tracing startup creates `lens_feedback` in the configured ClickHouse database. No PostgreSQL feedback table or migration is needed.
- Agent tracing must be enabled. Every operation rechecks the existing trace-read permissions; supplying an arbitrary trace ID or feedback ID does not grant access. A span must belong to the selected trace.
- Authenticate with a LiteLLM key belonging to a user allowed to read that trace. Read-only users can view ratings but cannot submit, edit, or delete them. Only the rating's author may change or delete it.
- Keep API keys on your backend. Browser-safe presigned feedback tokens and delegated end-user identities are **not** part of this API. Multiple end users behind the same authenticated LiteLLM user share one rating; do not treat an arbitrary client-supplied user ID as authenticated identity.

## Create or replace your rating

```bash
curl --fail-with-body "$LITELLM_URL/v1/feedback" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "trace_id": "<trace-id>",
    "trace_ref": "<trace-ref-from-trace-response>",
    "span_id": "<response-span-id>",
    "key": "usefulness",
    "value": 8,
    "comment": "Helpful, but missing an example"
  }'
```

The response includes `id`, the canonical trace reference, the submitted rating, and `created_at` / `updated_at`. Author identifiers and API key hashes are not exposed.

- `trace_id` and `value` are required. `key` defaults to `usefulness` and cannot be another criterion.
- Omit `span_id` (or send `""`) to rate the whole trace. Trace-level and individual span ratings have separate summaries.
- Copy `trace_ref` from the trace response when available. When omitted, the server resolves it using the caller's scope; ambiguous IDs are rejected rather than attaching feedback to the wrong tenant's trace.
- Values must be JSON integers: reject `-1`, `11`, fractional numbers, numeric strings, booleans, and null. Comments are limited to 4,000 characters.
- There is one logical rating per canonical trace/span and authenticated author. Repeating a POST, including concurrent retries, appends a version under the same ID rather than inflating counts. Different authors remain separate. Reads use `FINAL` to select the highest version without waiting for background merges. Concurrent writes resolve by timestamp plus a random tie-breaker; this is not transactional compare-and-swap. Keep replica clocks synchronized.
- Feedback cannot be submitted before trace ingestion becomes visible. A missing trace/span returns `404`; retry after ingestion completes. Unresolved identity returns `409`.

## Read and manage

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/v1/feedback` | Create or replace your rating |
| GET | `/v1/feedback?trace_id=...&trace_ref=...&span_id=...` | List ratings on that exact target |
| GET | `/v1/feedback/summary?trace_id=...&trace_ref=...&span_id=...` | Count, mean, all buckets 0–10, your rating, and `can_rate` |
| GET | `/v1/feedback/{id}` | Read a rating |
| PATCH | `/v1/feedback/{id}` | Replace your score and comment; target and author cannot change |
| DELETE | `/v1/feedback/{id}` | Remove your rating; returns `204` |

List responses contain `data` and `next_offset`. Use `limit` (1–100, default 50) and `offset` (default 0); `next_offset: null` means no next page. Results are ordered by stable feedback ID. Concurrent additions/deletions can shift offset pages.

PATCH takes `{"value": 10, "comment": "Now fully answered"}`. `value` is required; omitted `comment` clears the comment, as with POST.

An empty summary has `count: 0`, `average: null`, zeroes in every distribution bucket, and `mine: null`. A saved score of zero has `count: 1`, `average: 0`, and a count in bucket `"0"`. Writes are synchronous ClickHouse inserts. Read-after-write across independently lagging replicas is not guaranteed. Deletes append a tombstone and disappear from API reads without a blocking mutation.

Feedback does not extend trace retention. Once the trace is no longer accessible, its feedback cannot be read or changed through these endpoints. ClickHouse feedback retention is independent; include `lens_feedback` in your organization's data deletion/retention procedures. Tombstones are retained to prevent older versions from reappearing; an API delete is not physical erasure of historical comment bytes.
