# Project-filtered spend logs: delivery and acceptance

## Problem and evidence

A read-only usage query on a deployment reporting v1.104.1 took 42.6 seconds, fetched 46,349 raw records over 48 requests, and consumed 425,741,630 decoded response bytes (425.7 MB). Only 22 records matched the application's registered project IDs. These are observations from one query, not a controlled latency benchmark or wire-transfer measurement

Team, time-window and optional user filters were already applied. Increasing page size to 1,000 reduced round trips without reducing the full result set. Raising the client response limit avoided oversized-page failures but did not solve over-fetching. Positive and negative API controls showed that the deployed endpoint ignored project_id and project_ids. The search parameter did not search project metadata

Filtering by current keys or models could omit historical records. Daily aggregates could not preserve request-level identity, deduplication and missing/null/zero cache coverage. The proposed scope therefore remains exact project filtering of raw spend logs

## Proposed API contract

Accept repeated project_ids on GET /spend/logs/v2 in ungrouped mode. Match exact JSON strings in metadata.user_api_key_project_id, intersected with existing authorization, team, user, time and other filters. Apply the predicate before both counting and pagination, using bound SQL parameters. Omission preserves the existing behavior

Example request using synthetic identifiers:

```http
GET /spend/logs/v2?team_id=example-team&start_date=2026-01-01%2000:00:00&end_date=2026-01-02%2000:00:00&page=1&page_size=1000&group_by_session=false&project_ids=example-project-a&project_ids=example-project-b
```

Every filtered response, including empty pages, includes:

```json
{"project_filter":{"version":1,"project_ids":["example-project-a","example-project-b"]}}
```

The marker lets clients detect older servers that silently ignore unknown parameters. It does not grant access or replace authorization and local validation. Blank identifiers and incompatible grouped/UI/cursor modes return HTTP 400

## Minimal integration

1. Review the source change and integration tests against the target LiteLLM revision. The original release patch was validated at v1.104.1, commit 48963be81b8a1844e09a7769c940bc25d5c66d54. Do not assume private builds match that source
2. Build and deploy the patched source through the normal release pipeline. If the original build repository is unavailable, a derived image is an option only after verifying the actual image digest, Python module origin, file hash and startup/mount configuration. A UI URL or version label alone is insufficient
3. Keep the client's new source-filter opt-in disabled until direct positive and negative read-only controls pass against the deployed source
4. Derive project_ids from the server-side registered bindings and retain the existing team/user/time scope. Send repeated parameters on every page, including the terminal request
5. Require an exact supported marker on every response. Reject missing or mismatched markers instead of silently scanning everything. Retain local authorization, project validation, deduplication and cache-value handling
6. Roll back client usage by omitting the new parameter. This restores the original query semantics and its original over-fetching cost

No schema migration or index is included. Returning fewer rows does not itself prove fewer database rows are scanned. Query-plan and index work can follow measured results

## Acceptance method

Use existing historical records only, with the same identity, team, fixed UTC interval and registered project set. Read the baseline to an empty page, then apply the exact project predicate locally. Read the filtered endpoint to an empty page and compare the complete normalized records, request counts, input/output tokens, spend, cache tokens, coverage denominators and missing/null/zero classifications

A matching project must return its records; a nonexistent project must return an empty result with the marker. Test multiple projects, member scope, empty results, forbidden teams, omitted parameters and invalid modes. Preserve the existing source date inclusivity and any client half-open boundary checks

Record total duration, source duration, page count, returned rows and decoded bytes separately. Do not rely on a capped total_pages value. After filtering, the client cannot infer how many excluded records remain in the source

## Validation boundaries

For the upstream PR, the patch was applied to main at 66c8f5b1a9cefaa7c55b758f86f4ab552f2e859e in an isolated branch. All 17 integration cases passed against a real local HTTP proxy and PostgreSQL, and all 248 tests in test_spend_management_endpoints.py passed. Scoped Ruff checks and formatting passed. Full repository CI and production performance remain unverified

The unchanged v1.104.1 patch passed 17 local integration tests against real HTTP and PostgreSQL, plus 88 existing log/authorization tests. A synthetic 46,349-row fixture produced the same 22 complete records with 2 requests instead of 48, including the terminal empty page. Removing the new SQL predicate caused the targeted regression test to fail

Local curl observations on synthetic data were 1,000 rows / 843,090 bytes for the unfiltered first page and 22 rows / 18,585 bytes for the filtered page. These figures are not production speedup claims. No paid model requests were needed

The public PR status, its base revision and any additional validation are recorded in the PR. Upstream CI, merge, deployment and production performance acceptance are separate gates

## Running the integration contract

Follow the owned-service setup in [the integration README](../README.md). Against an isolated test deployment with PostgreSQL and the configured test identity, run `python tests/integration/run.py accounting tests/integration/spend/test_project_filtered_spend_logs.py`. The fixture inserts synthetic rows directly and cleans up its own rows. It does not invoke a model

The integration directory runs in the upstream main-branch CircleCI groups, not automatically as a new PR-specific job. Run this focused contract locally before relying on merge-time execution
