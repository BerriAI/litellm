# PR2 live replay, heads 7f7885ab86ebe27e1343c870970c220d76776722 and 97a66c4d850960e6aa53165c3daa9d7265e32398 vs baseline a2e1e62893de4ddcf3a26155d8cda15abb955b88

Verification artifacts only, these files are published on a separate evidence branch and are not part of either feature PR. Both sides ran from source with two proxy workers and one dedicated Redis (port 6392).

## Labels
- REAL: mode `real`. relay.py forwards the request unchanged to https://api.anthropic.com/v1/messages (anthropic/claude-sonnet-5-5). Exactly two REAL requests were made at 97a66c4d850. The final test-only tip carries that evidence as explained below.
- FIXTURE: every other mode. relay.py returns a canned Anthropic reply: write (cache_creation 2048), hit (cache_read 2048), miss (0/0), nousage (no cache counters), fail (HTTP 500), stream (canned 0/0 SSE). FIXTURE replies are not provider misses.

## Env var names (values not included)
ANTHROPIC_API_KEY, DATABASE_URL. LITELLM_LICENSE unset. LITELLM_MASTER_KEY and LITELLM_SALT_KEY are disposable literals in start.sh.

## Commands
```
redis-server --port 6392 --appendonly no --dbfilename pr2n.rdb --dir <dir> --daemonize yes
python -m uvicorn relay:app --port 4395
python gen_cfg.py ON 1 1 0          # ON.yaml: prompt_cache_alerts true, Redis on, original accounts
# from the worktree at the SHA under test (start.sh <worktree> ON.yaml <db> <log>):
python litellm/proxy/proxy_cli.py --config ON.yaml --port 4396 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver
RUN=<id> python drv2.py <db> head.jsonl <step>:<deployment>      # one step per call
bash state.sh head.jsonl "<step>"                                # dumps detector Redis keys after the step
```
relay.py, gen_cfg.py and start.sh are unchanged from the blobs uploaded for 0f89cd086b5. drv2.py and state.sh are in this bundle.

## Head sequence at 97a66c4d850 (head_97a66c4.jsonl has every step and the Redis state read after it)
1. real_proof:pc-a misses3:pc-a misses6:pc-a
2. fx_incident:pc-b break_fail:pc-c break_hit:pc-d break_nousage:pc-e break_stream:pc-f never_proven:pc-g
3. fx_proof_2miss:pc-h nolog_miss2:pc-h nolog_hit:pc-h miss1:pc-h
4. dismiss:pc-a (PATCH /alerts/10 status dismissed, HTTP 200)
5. In parallel on pc-i: fx_proof, 306 s idle, misses3, fx_hit, misses3, fx_proof, misses3
6. Stop both workers, `redis-cli shutdown save`, start Redis, `redis-cli flushall` (dbsize 0), start both workers: fx_proof:pc-a misses3:pc-a

## Baseline sequence (base_a2e1e62.jsonl), a2e1e62893d, same ON.yaml, same Redis
fx_incident:pc-b only, FIXTURE. No REAL request was made on the baseline.

## Results (rows = prompt_cache rows for that deployment in LiteLLM_Alert)
| Case | Upstream | Result | Completion signal |
|---|---|---|---|
| write then read | REAL | r0/w4634 then r4634/w0, 0 rows | Redis state hash `tokens 4634` |
| 3 zero-usage replies after it | FIXTURE | row 10 open, consecutive_misses 3, proven_write_tokens 4634 | the row, quiet key created |
| 6 more zero-usage replies | FIXTURE | row 10 unchanged (same updated_at) | the row |
| write, hit, 3 zero-usage | FIXTURE | row 11 open, count 3, tokens 2048 | the row |
| write, 2 miss, HTTP 500, 2 miss | FIXTURE | 0 rows | deployment reset key present after the step |
| write, 2 miss, hit, 2 miss | FIXTURE | 0 rows | misses zset back at 2 after the hit, not 4 |
| write, 2 miss, reply without usage, 2 miss | FIXTURE | 0 rows | deployment reset key present |
| write, 2 miss, stream, 2 miss | FIXTURE | 0 rows | deployment reset key present |
| 5 zero-usage, never proven | FIXTURE | 0 rows | no new detector key appeared for the step (key counts identical before and after) |
| write, 2 miss, 2 no-log miss, no-log hit, 1 miss | FIXTURE | row 12 opens on the last miss with count 3 | misses zset stayed at 2 through the three no-log calls |
| dismiss row 10; restart workers and Redis with flushall; fresh proof; 3 zero-usage | FIXTURE | row 10 still dismissed, no new row | quiet key re-created |
| proof, 306 s idle, 3 miss, hit, 3 miss | FIXTURE | 0 rows | row check only (observational) |
| fresh proof, 3 miss | FIXTURE | row 13 open | the row |
| baseline a2e1e62893d: write, hit, 3 zero-usage | FIXTURE | 0 rows, 0 prompt_cache rows in the baseline database | detector key count in Redis unchanged (2 before, 2 after) |

Limits: driver steps wait 0.5 s between requests and 3 s before reading rows. Where the completion signal is "row check only", the absence of a row is observational and rests on the unit and real-Redis tests. The expiry code is unchanged since 5796f53c4cb.

## Final tip 7f7885ab86ebe27e1343c870970c220d76776722 (head_7f7885a.jsonl)
`git diff 97a66c4d850 7f7885ab86e` changes only tests/unit/proxy/proxy_server/test_lifecycle.py (3 lines in, 3 out), so everything under litellm/ is byte-identical and the REAL write/read pair above carries. No REAL request was made at 7f7885ab86e.
Checked out 7f7885ab86e, same ON.yaml, database, Redis and two workers. All FIXTURE:
| Case | Result | Completion signal |
|---|---|---|
| write, hit, 3 zero-usage on pc-j (no earlier row) | row 17 open, count 3, tokens 2048 | the row |
| write, hit, 3 zero-usage on pc-b (row 11 from 97a66c4d850 still open) | row 11 unchanged, no second row | the row |
| 6 more zero-usage on pc-b | row 11 unchanged | the row |
| write, 2 miss, hit, 2 miss on pc-d | 0 rows | misses zset in the state dump |
| write, 2 miss, HTTP 500, 2 miss on pc-c | 0 rows | reset key in the state dump |
| PATCH /alerts/11 dismissed (HTTP 200), then 3 zero-usage on pc-b | row 11 dismissed, no new row | the row |
| write, 2 miss, 2 no-log miss, no-log hit, 1 miss on pc-h (row 12 from 97a66c4d850 still open) | row 12 unchanged, no second row | misses zset 2 after the proof step, still 2 after the no-log misses and after the no-log hit; after the last miss the zset is gone and a new quiet key exists |
