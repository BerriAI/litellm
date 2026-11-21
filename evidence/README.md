# Issue 49929 verification proof

Cache hit must return the stored finish_reason for an Anthropic response with a tool_use block and stop_reason end_turn.

- Baseline: 6bcc7ff6b9ea7f21c6801cf999429a04fcc78664 (main)
- Final tip: 016ff67b855f8dfe0e23d6cf6b847c4d5a727c11 (litellm_preserve_cached_finish_reason_49929). Production code is identical from d9f704e2 onward, later commits are test-only.
- Same venv and native build for both. The litellm/rust tree, uv.lock and pyproject.toml hashes are identical between the two commits.
- No tracked file was changed in either checkout by the verification.

## Setup

Env names used: ANTHROPIC_API_KEY (live smoke and real Anthropic proxy rows), LITELLM_MASTER_KEY (proxy), LITELLM_LOCAL_MODEL_COST_MAP=True. DATABASE_URL and DIRECT_URL are unset for every run.

    git worktree add --detach <head> 016ff67b855f8dfe0e23d6cf6b847c4d5a727c11
    ln -s <base>/litellm/rust_bridge/_native.abi3.so <head>/litellm/rust_bridge/_native.abi3.so
    # SDK: original issue script, compat matrix, finish_details, bridge probe, live smoke
    scripts/run_sdk.sh <checkout> <out dir>
    python3 scripts/compare.py <baseline out dir> <head out dir>
    # Proxy: dedicated Redis and the offline mock upstream, then one leg per checkout
    redis-server --port 6390 --bind 127.0.0.1 --save "" --appendonly no
    python scripts/mock_anthropic.py 8791
    scripts/run_proxy_leg.sh <checkout> <port> <out dir>
    # Tests with coverage, lint gates, basedpyright
    scripts/run_tests.sh <head> <out dir>
    scripts/run_gates.sh <head> 6bcc7ff6b9ea7f21c6801cf999429a04fcc78664 <out dir>
    scripts/run_typecheck.sh <head> 6bcc7ff6b9ea7f21c6801cf999429a04fcc78664 <out dir>

The proxy is started by scripts/start_proxy.sh as
`python litellm/proxy/proxy_cli.py --config scripts/config.yaml --port <port> --num_workers 2`.
Dropped from the repo's standard QA command: `--reload` (so both workers serve), `--detailed_debug`, the DB and `--use_v2_migration_resolver` (the cache path needs no DB).

The scripts never use elapsed time as evidence. A cache write is awaited through litellm's pending-write tasks and then observed in the store (bounded polling), and every same-mode second call asserts one provider request, the same response id and, for non-stream calls, the cache-hit marker. scripts/repro_issue49929.py is the issue's script, unmodified (sha256 8e039e500a002ad2871cc3f788e1f80a60394b9f4045a4852c8d66a48da58dc2).

## Original issue script

| Line | 6bcc7ff6 | 016ff67b |
|---|---|---|
| stream=False call 1 | stop | stop |
| stream=False call 2 (cache hit) | tool_calls | stop |
| stream=True call 1 (last chunk, stream_chunk_builder) | tool_calls, tool_calls | same |
| stream=True call 2 (last chunk, stream_chunk_builder) | tool_calls, tool_calls | same |
| requests that reached the mock | 2 | 2 |

## SDK compatibility matrix (final-016ff67b/compare_vs_baseline.txt)

143 case lines, 18 differ, all with an explicit stored `stop` and tool calls.

| Case | 6bcc7ff6 | 016ff67b |
|---|---|---|
| Explicit stop + tools, plain sync and async | tool_calls | stop |
| Explicit stop + tools, stream sync and async (last chunk and builder) | tool_calls | stop |
| Explicit stop + tools with finish_details stop or length, all four modes | tool_calls | stop |
| Chat-shaped entry at the Responses bridge cache site, stored stop | tool_calls | stop |
| Missing or null finish_reason, with or without finish_details, tools, all modes | unchanged | unchanged |
| Stored tool_calls, stored length, plain text stop and length | unchanged | unchanged |
| Real Anthropic claude-sonnet-5-5, tool use and text, plain and stream, miss and hit | tool_calls / stop | identical |

## Proxy, 2 workers, Redis cache (proxy_summary.txt in each directory)

| Case | 6bcc7ff6 | 016ff67b |
|---|---|---|
| /v1/chat/completions mock tool_use + end_turn, plain: miss, hit, upstream requests | stop, tool_calls, 1 | stop, stop, 1 |
| Same, stream | tool_calls, tool_calls, 1 | same |
| Mock tool_use + tool_use, text end_turn, text max_tokens, plain and stream | hit equals miss | same |
| /v1/responses mock tool use and text | completed, same output types | same |
| Real Anthropic chat tool use and text, plain and stream, and responses text | hit equals miss | same |

Two serving worker PIDs were observed in both recorded legs. The baseline log also recorded one early worker death. The final quiet-machine probe kept both workers alive with zero deaths. Earlier candidate runs concurrent with gates lost workers; that infrastructure behavior was not root-caused and is not presented as successful two-worker proof.

## Tests, coverage, gates at 016ff67b

- Mapped files (test_caching_handler.py, test_convert_dict_to_response.py, test_convert_to_streaming_response.py, test_streaming_handler.py): 941 tests, 940 passed, 1 failed.
- The failure is the pre-existing `test_sync_cache_hits_reach_sync_custom_logger[text_completion-request_kwargs2]`, not touched by the branch, a text completion with no tool calls. It passed in three earlier runs of the same production code, in 5 targeted reruns and in a whole-file rerun. Classified as an unrelated load-sensitive flake. It was not run on baseline under the same load.
- Added tests: 56 cases pass at the tip. Against 6bcc7ff6 source 8 fail (the explicit stored stop cases) and 48 pass (added_tests_baseline_red.txt).
- Changed executable lines covered 13 of 13, branch arcs from changed lines 6 of 6 (tests_and_coverage.txt).
- Gates, all exit 0 (gates.txt): uv lock check, ruff format on changed source, MCP operation boundary, full-source Ruff, test-tree Ruff, ruff_strict_gate, type_discipline_gate, test_quality_gate, openapi_docs_gate, type_check_gate (basedpyright), circular imports, both migration checks.

## Limits

- Real Anthropic returned tool_calls on the tool-use miss, so the tool_use + end_turn shape did not occur live. The live rows are a regression check, and the fix itself is shown with the issue's offline mock, through the SDK and the proxy.
- Forced tool choice was not exercised live: LiteLLM rejects it for claude-sonnet-5-5 (UnsupportedParamsError).
- Seeded stored values (explicit stop in a stream entry, missing, null, finish_details) are shown through the SDK only. With 2 proxy workers a rewritten Redis entry was read on only one of two hits, so seeded proxy rows were dropped.
- A normal Anthropic responses() cache hit is served from the Responses-shaped entry and reaches neither converter site. The bridge site was reached only with a chat-shaped entry seeded under the Responses key.
- Plain and stream calls do not share a cache entry (a plain miss followed by a stream call reaches the provider twice).
- Coverage is line and arc based, so multi-term conditions are not proven term by term by coverage, and the SDK evidence matrix uses one choice. The committed cache round-trip unit tests cover two choices.
- The baseline-red run used the 4770473d test file, whose later change at 016ff67b only restructures the no-choices test.
- Gates were run locally in the locked env against merge-base 6bcc7ff6, not in GitHub Actions.

## Evidence files

The scripts directory contains the exact final verification fixtures. Full private logs are retained in the verifier session. Compact measured results are included here.
