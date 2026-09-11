# LIT-4896 verification

Baseline: `7419a536ad6857165b28b9b545a62d5916f696d2`

Fix: `ca03c889c9b72ec47a5d4f604626ac0b7cfdfadd`

## Runtime setup

1. Use isolated source checkouts at the two commits and set `LITELLM_SOURCE` to the checkout being tested
2. Build the repository Dockerfile once. The recorded runs reuse the successful `litellm-4896:8c82c325ac` dependency/Rust/dashboard layers, then `build_source_image.py COMMIT` copies the complete tracked Python package from each commit into its own image. No dependency or compiled UI behavior changed; the generated dashboard schema differs only in two descriptions
3. Run `verify_runtime.py COMMIT before|after` to compare SHA-256 hashes of all three touched runtime modules with `git show`. Recorded image IDs and hashes are in the runtime JSON files
4. Create a private `.env` containing POSTGRES_PASSWORD, LITELLM_MASTER_KEY, and LITELLM_SALT_KEY. Install the harness requirements in an isolated Python environment and use an authenticated `gh` CLI for the real upstream credential
5. Start `docker compose up -d db spec-host`, then `python switch_runtime.py COMMIT`. Gateway: http://localhost:44896, spec host: http://localhost:44897. Resource ceilings total 1.5 CPU and 1408 MiB RAM
6. Register the scenarios through `seed.py`, `seed_health.py`, and `seed_extra.py` on a fresh database. Reuse the database across before/after runs. `health-servers.json` records the actual IDs from these runs
7. Open http://localhost:44896/ui/mcp-servers/ and log in as admin with the private master key. Connect a valid caller credential on remote_none, remote_bearer, byok_default, and byok_bearer. The latter two use a local specification file
8. Use the public pinned specification in github-openapi.json and the public native endpoint https://mcp.deepwiki.com/mcp. The spec-host serves actual HTTP 404, invalid JSON, delayed, oversized, and forcibly compressed responses after startup. The registry reload interval is 3600 seconds so existing failing-spec servers remain registered throughout screenshots

## Matching before/after actions

1. On the MCP Servers page, use the remote_none card menu > Test Connection. Inspect all ten badges, then click/hover each failure or Unknown badge for its error tooltip. See before/after-health.png and the six matching tooltip images
2. Query each server directly: `curl -sS -H "Authorization: Bearer $LITELLM_MASTER_KEY" http://localhost:44896/v1/mcp/server/SERVER_ID`. The health JSON files retain exact URLs, HTTP status, health status, errors, duration, and source commit. `capture_health.py before|after COMMIT` performs these curl requests without exposing keys
3. Open http://localhost:44896/ui/playground/, select Current UI Session, Endpoint Type `/mcp-rest/tools/call`, server remote_none, tool getauthenticateduser, then Send. Both commits return HTTP 200 and `isError: false`. See tool screenshots and sanitized response summaries
4. For each of the four OpenAPI cases, remove the caller credential, save an invalid one, then save a valid one. List tools and call getauthenticateduser after each change. Missing keys return HTTP 401, invalid keys return tool `isError: true` with upstream 401, valid keys return HTTP 200 and `isError: false`
5. List native_http tools and call read_wiki_structure with `{"repoName":"BerriAI/litellm"}`. Both commits list three tools and return successful content. The 13 before/after control records are identical
6. Stop this project with `docker compose stop` and close the dedicated test browser. Both are stopped after the recorded run

## Results

Remote specifications with None or Bearer auth change from Unhealthy/Unknown to Healthy. Local specs return Unknown with an explicit protocol-probe explanation. Remote 404, invalid JSON, and timeout failures become actionable Unhealthy errors. Responses over 10 MiB or forced compression return Unknown, with the refusal explained. Native health remains Healthy, and tool execution and credential enforcement remain unchanged

Healthy establishes specification availability and parsing only. It does not establish operation availability or validate caller credentials. Results are cached for 30 seconds; concurrent checks share one probe. Cancelled results are never cached

## Local validation

Run the three mapped test files with branch coverage:

```sh
.venv/bin/python -m pytest \
 tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_manager.py \
 tests/test_litellm/proxy/_experimental/mcp_server/test_openapi_to_mcp_generator.py \
 tests/test_litellm/llms/custom_httpx/test_http_handler.py \
 --cov=litellm.proxy._experimental.mcp_server.mcp_server_manager \
 --cov=litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator \
 --cov=litellm.llms.custom_httpx.http_handler \
 --cov-branch --cov-report=xml:coverage.xml --cov-report=json:coverage.json -q
make lint
cd ui/litellm-dashboard
NODE_OPTIONS=--max-old-space-size=3072 RAYON_NUM_THREADS=2 VIPS_CONCURRENCY=2 npm run build
```

774 tests passed. All canonical backend lint gates passed without budget changes. Six focused dashboard component tests, four type tests, and the dashboard build passed

Local changed executable-line coverage is 71/71 (100%). Touched-function branch coverage is 31/32 (96.875%): the uncovered branch is the unchanged local-file FileNotFound branch in the loader, which local-file health never enters. Other uncovered touched-function lines belong to unchanged native MCP handling. New executable branches are covered. The coverage JSON/XML and changed-coverage report retain the details

Completed current-tip Codecov reports 71/71 patch lines covered (100%), separately from repository-wide coverage of 80.54%. All CI upload jobs completed before the report was fetched

41 selected regressions fail against the merge base. The cancellation-cache defect was introduced during review, reproduced against da1dfcdb24 using a real HTTP server, and fixed at the final commit: one request and Unknown/Unknown before, two requests and Unknown/Healthy after. Both already-waiting and newly-arriving followers have assertions in the mapped manager tests

The unbounded-loading review concern was reproduced before the bounding change with three complete 12 MiB responses. At the final commit, concurrent probes issue one request and reject from the size header before consuming the body. The `bytes_sent` diagnostic counts completed whole-response writes, not precise network bytes

## Review status

Current commit ca03c889c9: Greptile 5/5 with no actionable findings; Veria reports no security issues; Cursor Bugbot reports no issues. All three prior findings are resolved and retained with their reproduction evidence. The author confirmed the existing CLA signature as joshua-berri; no automated CLA check was posted
