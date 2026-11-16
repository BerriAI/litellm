# PR2 E2E replay, head 0f89cd086b5a77a59c7631292b228959974f5986 vs baseline f594c55059425e024a12200e8c3d0cbf00814b8e

Verification artifacts only, none of these files are in the repository.

## Labels
- REAL: `MODE:real`, the relay forwards the request unchanged to https://api.anthropic.com/v1/messages (model anthropic/claude-sonnet-5-5).
- FIXTURE: every other mode. relay.py returns a canned Anthropic reply: write (cache_creation 2048), hit (cache_read 2048), miss (0/0), nousage (no cache counters), fail (HTTP 500), stream (canned 0/0 SSE).

## Env var names (values not included)
ANTHROPIC_API_KEY, DATABASE_URL (Postgres, one empty migrated database per side). LITELLM_LICENSE unset.
LITELLM_MASTER_KEY and LITELLM_SALT_KEY are disposable literals set in start.sh. `fixture-account-1/2` and `unused` in gen_cfg.py are fixture strings, not credentials.

## Commands
```
redis-server --port 6392 --appendonly no --dbfilename pr2n.rdb --dir <dir> --daemonize yes      # redis.sh up
python -m uvicorn relay:app --port 4395                                                          # relay
python gen_cfg.py ON 1 1 0
python gen_cfg.py ON2 1 1 1
python gen_cfg.py OFF 0 1 0
python gen_cfg.py NOREDIS 1 0 0
# start.sh <worktree> <config> <db> <log>, run from the worktree at the SHA under test:
python litellm/proxy/proxy_cli.py --config <config> --port 4396 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver
RUN=<id> python drv.py <db> <out.jsonl> <step>:<deployment> ...
```
ON.yaml: 22 Anthropic deployments pc-a..pc-v pointing at the relay plus pc-oai, router_settings redis and general_settings.coordination_redis on 127.0.0.1:6392, general_settings.prompt_cache_alerts: true.
ON2.yaml: same, pc-m api_base changed, pc-n api_key changed. OFF.yaml: prompt_cache_alerts absent. NOREDIS.yaml: no Redis settings.
Two proxy worker PIDs served on both sides and the "workers flag is ignored" warning did not appear.

## Sequence as run (head.jsonl holds every step, marker lines record each environment change)
1. ON.yaml: real_proof:pc-a misses3:pc-a show:pc-a misses6:pc-a fx_incident:pc-b break_fail:pc-c break_hit:pc-d break_nousage:pc-e break_stream:pc-f never_proven:pc-g
2. surface_messages:pc-h surface_responses:pc-i all_stream:pc-j two_breakpoints:pc-k ttl_1h:pc-l provider_openai:pc-oai dismiss:pc-a fx_proof_2miss:pc-m fx_proof_2miss:pc-n fx_proof_2miss:pc-o
3. stop proxy, `redis-cli shutdown save`, start Redis, start proxy with ON2.yaml: misses3:pc-a miss1:pc-o miss2:pc-m miss2:pc-n fx_proof_2miss:pc-m miss1:pc-m fx_proof_2miss:pc-p
4. `redis-cli flushall`: miss2:pc-p fx_proof_2miss:pc-q
5. Redis stopped: fx_incident:pc-r. Redis started: miss2:pc-r, fx_incident:pc-s (breaker still open), after 45 s fx_incident:pc-t
6. `alter table "LiteLLM_Alert" rename to "LiteLLM_Alert_hidden"`: fx_incident:pc-u. Rename back: miss2:pc-u
7. OFF.yaml: fx_incident:pc-v. NOREDIS.yaml: fx_incident:pc-v (new RUN id)
8. Gap run, new RUN id, ON.yaml: table renamed away, fx_incident:pc-u, renamed back, miss1:pc-u x3. Then fx_proof:pc-a misses3:pc-a. Then flushall, fx_proof:pc-a misses3:pc-a
Baseline f594c55 ran step 1 up to break_fail only (base.jsonl).

## Results (rows = prompt_cache rows for that deployment in LiteLLM_Alert)
| Case | Upstream | Result at head |
|---|---|---|
| real write 4632 then read 4632 | REAL | 0 rows |
| + 3 misses | REAL proof, FIXTURE misses | 1 row open, consecutive_misses 3, proven_write_tokens 4632, ttl_seconds 300, provider anthropic, resource_type deployment |
| + 6 misses | FIXTURE | still 1 row, updated_at unchanged |
| write, hit, 3 misses | FIXTURE | 1 row |
| HTTP 500 / hit / missing usage / streamed request between misses 2 and 3 | FIXTURE | 0 rows each |
| 5 misses, never proven | FIXTURE | 0 rows |
| /v1/messages, /v1/responses, all streamed, two breakpoints, 1h TTL, OpenAI | FIXTURE | 0 rows each |
| PATCH dismiss alert 1 | n/a | 200, dismissed |
| after Redis+proxy restart: 3 misses on dismissed pc-a | FIXTURE | row still dismissed, no new row |
| proof + 2 misses before restart, 1 miss after (pc-o) | FIXTURE | 1 row |
| api_base changed across restart (pc-m): 2 misses | FIXTURE | 0 rows, then fresh write + 3 misses opened 1 row |
| api_key changed across restart (pc-n): 2 misses | FIXTURE | 0 rows |
| FLUSHALL after proof + 2 misses, then 2 misses | FIXTURE | 0 rows |
| Redis down: write, hit, 3 misses | FIXTURE | all 200, 0 rows, log "prompt cache alert streak unavailable" |
| Redis back, breaker closed: write, hit, 3 misses | FIXTURE | 1 row |
| alert table missing: write, hit, 3 misses | FIXTURE | all 200, 0 rows, log "prompt cache alert not saved (TableNotFoundError)" |
| table restored: miss, miss, miss | FIXTURE | 0 rows, 0 rows, then 1 row (id 6, consecutive_misses 3) |
| dismissed pc-a, Redis state kept: fresh proof + 3 misses | FIXTURE | same row id 1 still dismissed, no open row |
| dismissed pc-a, after FLUSHALL: fresh proof + 3 misses | FIXTURE | same row id 1 still dismissed, no open row, 6 rows total unchanged, 1 quiet key re-created |
| prompt_cache_alerts absent | FIXTURE | all 200, 0 rows, detector Redis keys 8 before and 8 after |
| enabled, no Redis configured | FIXTURE | all 200, 0 rows |
| Baseline f594c55: real write/read, 3 misses, fixture incident, error break | REAL + FIXTURE | all 200 (except the injected 500), 0 rows, 0 detector keys |

## Not exercised live
None-client reload gap, 5-minute proof expiry, expiry of the one-day quiet period.

## Harness mistakes recorded in head.jsonl (not product behaviour)
A first table rename failed on a psql role error (CORRECTION marker). One NOREDIS run used pc-w, which is not in the config, and returned 400 (CORRECTION marker).

## Review outcome

PR2 is paused and must not merge. See GREPTILE_REPRO_0f89cd0.md for the three confirmed false-incident paths and the limitations of the live negative checks
