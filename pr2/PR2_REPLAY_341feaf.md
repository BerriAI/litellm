# PR #51973 prompt-cache advisory: verification replay at 341feafb5fc

Head `341feafb5fc2d8a3d72adf5873df6982d4db77da`, base `113766d4b90931d7f696576ef56c6721fb424f74` (main). Verification only, no source edits.

`341feafb5fc` differs from `46c231d2bd3` only in `tests/integration/caching/test_prompt_cache_alert_streaks_real_io.py` (4 lines in, 3 out). The eight-commit patch is otherwise line-identical to `a2e1e62893d..7f7885ab86e`.

## Runtime

Tests and proxies ran in one container, `ubuntu@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8` (glibc 2.39), host network, logs capped at 10m x 3. The host interpreter (Python 3.12.16), the repo venv and Node v24.22.3 were bind-mounted read-only. Three apt packages were added inside the container: `libsqlite3-0`, `libreadline8t64`, `ca-certificates`.

The native module is `_native.abi3.so` (SHA256 `ad25938fc16388592b2f5f14d4b1fdc3d46d57c4e7e31a8f6c669209d5575b0e`) from CI artifact 14279432858 (run 43974182323, zip SHA256 `43828a78a0a941dc887519b5078415492fb1d4f2f7957ef40ebfca5662f8f6fa`). CI restored it from cache key `Linux-rust-bridge-editable-46e9a1d6c779ae27daf33b919477300dbf9bedf19d3cc8bbc208ec7cd16cbbd7`, it was not compiled in that run. The six paths the restore action hashes (`litellm/rust/Cargo.lock`, `litellm/rust/Cargo.toml`, `litellm/rust/crates/**`, `model_prices_and_context_window.json`, `pyproject.toml`, `litellm/packaging/litellm-proxy/pyproject.toml`) are byte-identical at `113766d4b90`, `46c231d2bd3` and `341feafb5fc` (1434 tracked files, empty `git diff`). A local reimplementation of `hashFiles` gives the same value for all three commits but does not reproduce the literal key, so the key value itself is unverified. The same `.so` served base and head.

## Tests and gates at 341feafb5fc

| Check | Result |
|---|---|
| 131 affected units, serial | 131 passed |
| 131 affected units, `-n 4` with coverage | 131 passed |
| Ordered pair (save-worker test, then registration test) | 2 passed |
| Real Redis, 19 cases | 19 passed |
| ruff format, MCP boundary, ruff check, ruff on tests, ruff strict, type discipline, test quality, unbounded-in-lists (host, against 113766d4b90) | exit 0 each |
| OpenAPI docs checker, base vs head | identical output, 168 OA001, 39 OA002, 224 OA003 on both |
| Changed-line coverage vs 113766d4b90 | 280/288 |

## Live flow

Two workers, router Redis and coordination Redis both on a dedicated Redis at 6392, Postgres database per leg. Model `anthropic/claude-sonnet-5-5`, the newest Anthropic Sonnet in `model_prices_and_context_window.json`. `real` requests reach Anthropic. Every other mode is a canned relay FIXTURE and is not provider evidence.

| Leg | Step | Upstream | Result | Completion signal |
|---|---|---|---|---|
| Base 113766d4b90 | write, read | real Anthropic | 4636 written, then 4636 read | response usage |
| Base | 3 zero-usage replies | fixture | 0 rows, alerts list 200 and empty | 0 detector keys in Redis |
| Head 341feafb5fc | write, read | real Anthropic | 4636 written, then 4636 read, 0 rows | Redis state hash `tokens 4636` |
| Head | 3 zero-usage replies | fixture | 1 open row, count 3, proven tokens 4636 | the row, quiet key set |
| Head | 6 more | fixture | same row, `updated_at` unchanged | the row |
| Head | dismiss | none | PATCH 200, row `dismissed` | the row |
| Head | fresh write, 3 zero-usage | fixture | row still dismissed, list empty | the row (quiet key still live) |
| Head | stop workers, `FLUSHALL` on 6392, start workers, fresh write, 3 zero-usage | fixture | row still dismissed, no new row, list empty | new quiet key appears after the third miss |

Relay log: 2 `anthropic` and 3 `fixture` entries for base, 2 `anthropic` and 17 `fixture` for head. Raw results are in `base_113766d.jsonl`, `head_341feaf.jsonl` and `relay_log.jsonl`.

## Not rerun on current main

Carried from `97a66c4d850` and `7f7885ab86e` by patch equivalence: 306 s expiry, hit / 500 / missing usage / stream between misses, `no-log`, never-proven. The 19 Redis cases cover the expiry boundary at this tip.

Not run locally at this tip: type check gate and BasedPyright comparison (both need a Rust build), and `openapi_docs_gate.py --base` itself (its base fallback does an editable install). With `--reload` both worker processes were present, but which worker served each request was not checked.

## Rerun

Env var names: `DATABASE_URL`, `ANTHROPIC_API_KEY`, `LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`, `PROXY_BASE_URL`, `PYTHONPATH`, `RUN`, `REDIS_HOST`, `REDIS_PORT`.

```bash
redis-server --port 6392 --save '' --appendonly no --daemonize yes
python -m uvicorn relay:app --host 127.0.0.1 --port 4395 --log-level warning &
python gen_cfg.py ON 1 1 0
createdb cz_rs_base_113766d; createdb cz_rs_head_341feaf
bash ctr_up.sh
export LITELLM_MASTER_KEY=sk-pr2-disposable-master RUN=rs$(date +%s)
bash start_ctr.sh $WT_BASE ON.yaml cz_rs_base_113766d proxy_base.log
python3 drv2.py cz_rs_base_113766d base.jsonl real_proof:pc-a misses3:pc-a show:pc-a
bash stop_ctr.sh; redis-cli -p 6392 flushall
bash start_ctr.sh $WT_HEAD ON.yaml cz_rs_head_341feaf proxy_head.log
for s in real_proof misses3 show misses6 dismiss fx_proof misses3 show; do python3 drv2.py cz_rs_head_341feaf head.jsonl $s:pc-a; bash state.sh head.jsonl $s; done
bash stop_ctr.sh; redis-cli -p 6392 flushall
bash start_ctr.sh $WT_HEAD ON.yaml cz_rs_head_341feaf proxy_head.log
for s in fx_proof misses3 show; do python3 drv2.py cz_rs_head_341feaf head.jsonl $s:pc-a; bash state.sh head.jsonl restart_$s; done
```

Proxy command inside the container, from the worktree root:

```bash
python litellm/proxy/proxy_cli.py --config ON.yaml --port 4396 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver
```

<details><summary>Config used for pc-a (ON.yaml, other deployments omitted)</summary>

```yaml
model_list:
  - model_name: pc-a
    litellm_params: {model: anthropic/claude-sonnet-5-5, api_key: os.environ/ANTHROPIC_API_KEY, api_base: http://127.0.0.1:4395/main}
    model_info: {id: pc-a}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  coordination_redis: {host: 127.0.0.1, port: 6392}
  prompt_cache_alerts: true
router_settings:
  num_retries: 0
  redis_host: 127.0.0.1
  redis_port: 6392
```
</details>

<details><summary>Client request (from drv2.py)</summary>

```python
prefix = f"[{RUN}-pc-a] " + "Reference notes for the assistant. " * 420
body = {"model": "pc-a", "max_tokens": 8, "messages": [
    {"role": "system", "content": [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}]},
    {"role": "user", "content": f"Reply ok. {uuid.uuid4().hex} MODE:{mode}"}]}
# POST http://127.0.0.1:4396/v1/chat/completions with Authorization: Bearer $LITELLM_MASTER_KEY
# mode "real" is forwarded to api.anthropic.com by relay.py, "write" / "miss" return canned usage
# dismissal: PATCH /alerts/<id> {"status": "dismissed"}; list: GET /alerts?type=prompt_cache
```
</details>

<details><summary>Test commands (in the container, worktree at 341feafb5fc)</summary>

```bash
T="tests/unit/integrations/prompt_cache_alerts tests/unit/proxy/proxy_server/test_lifecycle.py"
python -m pytest $T -q -p no:cacheprovider
python -m pytest $T -q -p no:cacheprovider -n 4 --cov=litellm --cov-branch --cov-report=
python -m coverage json -o coverage.json
REDIS_HOST=127.0.0.1 REDIS_PORT=6392 python -m pytest tests/integration/caching/test_prompt_cache_alert_streaks_real_io.py -q -p no:cacheprovider
python litellm/scripts/check_openapi_docs.py   # once per worktree, outputs compared with diff
python3 cov_diff.py 113766d4b90931d7f696576ef56c6721fb424f74 coverage.json
```
</details>

`relay.py`, `gen_cfg.py`, `drv2.py` and `state.sh` are the blobs published earlier. `ctr_up.sh`, `start_ctr.sh` and `stop_ctr.sh` are new with this bundle.
