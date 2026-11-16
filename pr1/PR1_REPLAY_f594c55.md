# PR1 live replay: baseline eb8a2d224e75593899f6af418c0aa5d3b3ce3060 vs head f594c55059425e024a12200e8c3d0cbf00814b8e

All alert rows are seeded fixtures (PR1 has no detector). No model calls. One Postgres database and one dedicated Redis (port 6393) shared by both SHAs, two proxy workers at each SHA, `LITELLM_LICENSE` unset. Dashboard is the production build (`npm run build`) of each SHA served through `LITELLM_UI_PATH`.

## Setup (identical at both SHAs, only the checkout and UI directory differ)

Env var names: `DATABASE_URL`, `MASTER_KEY`, `LITELLM_SALT_KEY`, `LITELLM_UI_PATH`.

config.yaml
```yaml
model_list:
  - model_name: fixture-model
    litellm_params: {model: openai/fixture-model, api_key: unused, api_base: http://127.0.0.1:9/v1}
general_settings:
  master_key: os.environ/MASTER_KEY
  coordination_redis: {host: 127.0.0.1, port: 6393}
router_settings:
  redis_host: 127.0.0.1
  redis_port: 6393
```

```bash
redis-server --port 6393 --save "" --daemonize yes
cd litellm/ui/litellm-dashboard && npm run build && export LITELLM_UI_PATH=$PWD/out && cd -
python litellm/proxy/proxy_cli.py --config config.yaml --port 4000 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
```
Two worker PIDs served at both SHAs and the `"workers" flag is ignored` warning did not appear.

## Seed step A, existing alert types (ids 1 to 5)
```sql
INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias) VALUES ('fixture-team-1', 'Fixture Team') ON CONFLICT DO NOTHING;
INSERT INTO "LiteLLM_DailyModelUsage" (date, model_group, model, custom_llm_provider, task_type, spend, request_count, successful_requests, failed_requests, total_response_time_ms, timed_requests, updated_at)
SELECT to_char(d, 'YYYY-MM-DD'), 'fixture-model', 'fixture-model', 'openai', 'chat',
  CASE WHEN d = date '2026-11-14' THEN 42.0 ELSE 10.0 END, 100, 95, 5,
  CASE WHEN d = date '2026-11-14' THEN 250000 ELSE 80000 END, 100, now()
FROM generate_series(date '2026-11-07', date '2026-11-14', interval '1 day') d ON CONFLICT DO NOTHING;
INSERT INTO "LiteLLM_Alert" (detection_key, type, resource_type, resource_id, window_start, window_end, evidence, created_at, updated_at) VALUES
 (encode(sha256('fixture-spend'), 'hex'), 'spend', 'gateway', 'gateway', '2026-11-14 00:00', '2026-11-15 00:00', '{"value": 42.0, "baseline": 10.0, "unit": "usd", "sample_size": 100}', '2026-11-15 00:10', '2026-11-15 00:10'),
 (encode(sha256('fixture-traffic'), 'hex'), 'traffic', 'team', 'fixture-team-1', '2026-11-14 00:00', '2026-11-15 00:00', '{"value": 900, "baseline": 300, "unit": "requests", "sample_size": 900}', '2026-11-15 00:11', '2026-11-15 00:11'),
 (encode(sha256('fixture-latency'), 'hex'), 'latency', 'model', 'fixture-model', '2026-11-14 00:00', '2026-11-15 00:00', '{"value": 2500, "baseline": 800, "unit": "milliseconds", "sample_size": 100}', '2026-11-15 00:12', '2026-11-15 00:12'),
 (encode(sha256('fixture-reliability'), 'hex'), 'reliability', 'user', 'fixture-user-1', '2026-11-14 00:00', '2026-11-15 00:00', '{"value": 0.4, "baseline": 0.05, "unit": "ratio", "sample_size": 200}', '2026-11-15 00:13', '2026-11-15 00:13'),
 (encode(sha256('fixture-deprecation'), 'hex'), 'model_deprecation', 'model', 'fixture-old-model', '2026-11-14 00:00', '2026-11-15 00:00', '{"deprecation_date": "2026-12-31", "replacement_model": "fixture-new-model"}', '2026-11-15 00:14', '2026-11-15 00:14');
UPDATE "LiteLLM_PerformanceAlertState" SET alerts_used = NULL WHERE id = 1;
```

## Seed step B, the prompt_cache row (id 7)
```sql
INSERT INTO "LiteLLM_Alert" (id, detection_key, type, resource_type, resource_id, window_start, window_end, evidence, created_at, updated_at) VALUES
 (7, '41c3f4d5575b4c099a141c97dfe1d4a04f47f3f759799183b3f4b97a4edb75b3', 'prompt_cache', 'deployment', 'fixture-deployment-1', '2026-11-15 01:00:00', '2026-11-15 01:00:03', '{"config_fingerprint": "fixture-config-1", "prefix_fingerprint": "fixture-prefix-1", "provider": "anthropic", "provider_model": "claude-sonnet-5-5", "consecutive_misses": 3, "proven_write_tokens": 2048, "ttl_seconds": 300}', '2026-11-15 01:01', '2026-11-15 01:01');
UPDATE "LiteLLM_PerformanceAlertState" SET alerts_used = NULL WHERE id = 1;
SELECT setval(pg_get_serial_sequence('"LiteLLM_Alert"', 'id'), 7);
```

## Status-change steps
1. Baseline, after step A: open `/ui/alerts?alert=2`, click Dismiss. Row 2 becomes `dismissed`, the Dismissed tab lists it with Reopen.
2. Step B applied. Baseline captures taken. Baseline stopped.
3. Head started on the same database. Captures taken.
4. Head: open `/ui/alerts?alert=7`, click Dismiss. Row 7 becomes `dismissed` (`updated_by=default_user_id`), page shows "Dismissed" and Reopen, Dismissed tab and `GET /alerts?status=dismissed` list ids 7 and 2.
5. Head: `PATCH /alerts/7 {"status":"open"}` returns 200 and row 7 is `open` again.

## API, same request at both SHAs with rows 1 to 5 and 7 present (row 2 dismissed)
`curl -H "Authorization: Bearer $MASTER_KEY" http://127.0.0.1:4000<path>`

| Path | Baseline | Head |
|---|---|---|
| `/alerts` | 500 | 200, ids 7, 5, 4, 3, 1 |
| `/alerts?status=open&type=performance` | 500 | 200, ids 4, 3, 1 (metric types only) |
| `/alerts?type=spend` | 200, id 1 | 200, body identical to baseline |
| `/alerts?type=model_deprecation` | 200, id 5 | 200, identical |
| `/alerts?type=prompt_cache` | 422 | 200, id 7 |
| `/alerts?status=dismissed` | 200, id 2 | 200, identical |
| `/alerts/1` (spend, 8 history points) | 200 | 200, identical |
| `/alerts/2`, `/alerts/3`, `/alerts/4`, `/alerts/5` | 200 | 200, identical |
| `/alerts/7` | 500 | 200, `type=prompt_cache`, `resource_type=deployment`, `history=null`, evidence as seeded |
| `/alerts/6` | 404 | 404 |

Baseline 500s are `pydantic ValidationError ... for Alert` in the proxy log. Head log has 0 tracebacks.

Before step B, baseline returned 200 on `/alerts` (5 rows) and on the performance filter (4 metric rows), and 404 on `/alerts/7`.

## Quota, `GET /alerts/monitoring` after `alerts_used` reset to NULL
| Rows | Baseline `alerts_used` | Head `alerts_used` |
|---|---|---|
| ids 1 to 5 | 4 | not rerun at head |
| ids 1 to 5 and 7 | 5 | 4 |

Head response: `{"enabled":true,"model_deprecations_enabled":true,"alerts_used":4,"alert_limit":5,"disabled_by_config":false}`.

## Daily digest, shipped `AlertRepository.summaries(0, latest_id)` against the live database
| | Baseline | Head |
|---|---|---|
| ids 1 to 5, all open | ids 4, 3, 2, 1 | not rerun at head |
| ids 1 to 5 and 7, row 2 dismissed | `ValidationError` on `type='prompt_cache'` | ids 4, 3, 1, each entry identical to the baseline entry |

## UI, 1440x1000, logged in as admin, red outline drawn in the page before capture
| URL | Baseline | Head |
|---|---|---|
| `/ui/alerts` | "Could not load performance alerts." | list with "Prompt cache usage not observed on fixture-deployment-1" first, then the three open metric alerts and the deprecation alert |
| `/ui/alerts?alert=7` | "Could not load this alert." | detail: deployment, model, 3 requests in a row, 2048 tokens, 5 minutes, miss window, scope note; no history chart |
| `/ui/alerts?alert=1` | spend detail with daily chart | same |
| `/ui/alerts?alert=5` | deprecation detail | same ("Daily history is unavailable", retires Dec 31, 2026) |

## Limits
- The baseline dashboard build produced complete output and was served, but its exit status was not captured. The head build exited 0.
- Screenshots show the product's "Environment-credential login is enabled" banner, version v1.111.0 and the generic "Account / Admin" footer. No license card, key or user data.
- The existing real-DB test `test_opt_out_is_shared_and_config_false_cannot_be_overridden` fails the same way at baseline and head (extra `model_deprecations_enabled` key). Not in this diff.
