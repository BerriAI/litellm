# Database upgrade performance

These opt-in tools prepare an unpartitioned, metadata-only dataset and qualify a disposable database upgrade. Snapshot provisioning and automatic stable-release resolution are not wired yet. They do not run in default E2E collection

## Seed

Apply the baseline release's migrations to an empty, disposable PostgreSQL database, stop all proxy writers, and set `DATABASE_URL`. Use a fixed UTC midnight before today for the anchor

```sh
uv run python tests/e2e/perf/database_seed.py \
  --baseline-version "$BASELINE_VERSION" \
  --anchor 2026-10-15T00:00:00Z \
  --manifest dataset-manifest.json \
  --confirm-disposable
```

The default target is 1,000,000,000,000 measured bytes in spend logs and their indexes. Daily aggregates and other tables take additional space. For a small validation, add `--target-bytes 100000000 --batch-rows 10000`. Reuse the identical arguments to resume. The seed rejects a different specification, migration history, generator, or a partitioned spend-log table

The checkpoint and manifest live in the database. Preserve them in the snapshot along with the PostgreSQL major version. Always restore a new disposable copy for an upgrade, never run against the saved baseline

## Upgrade

Start the baseline proxy against the restored database. Provide a separately addressed target deployment, initially stopped, using the same database and master key. Configure a working chat model on both proxies, enable spend logging, and keep `store_prompts_in_spend_logs=false`. Set `DATABASE_URL`, `LITELLM_MASTER_KEY`, and `E2E_PERF_TESTS=1`

Run from the repository root with `PYTHONPATH=.:tests/e2e`

```sh
PYTHONPATH=.:tests/e2e uv run python -m tests.e2e.perf.database_upgrade \
  --baseline-version "$BASELINE_VERSION" \
  --target-version "$TARGET_VERSION" \
  --stable-url "$STABLE_URL" --target-url "$TARGET_URL" \
  --model "$MODEL" \
  --target-index LiteLLM_SpendLogs_api_key_startTime_idx \
  --target-index LiteLLM_SpendLogs_litellm_call_id_idx \
  --output upgrade-results --confirm-disposable \
  -- kubectl apply -f target-proxy.yaml
```

`target-proxy.yaml` is the caller's normal serving-proxy deployment using the selected target image and the restored database. The command must initiate startup and return. Leave the baseline proxy running. Supply every request-log index name expected by the target release, not a list inferred from the baseline. The example names are the current request-log indexes. This runner targets normal serving-proxy startup, not the separate `--skip_server_startup` migration-job mode

The runner first validates the manifest and baseline version, benchmarks the real Logs and Usage APIs, and verifies baseline completions and their spend rows. It starts the upgrade command while sending baseline traffic and observing database locks. As soon as the target reports the requested version and a connected database, it sends target completions too

A separate target completion is sent while PostgreSQL reports an active expected index build. Its spend row must become visible before that build finishes. The runner waits for all expected indexes to become valid, then checks target traffic, new spend rows, existing-row fingerprints, migration history, and the same dashboard queries

`background_index_overlap` is `passed` only when the target request and its persisted spend row overlap a build. Pending work without that proof is `failed`. Already-complete indexes are reported as `no-pending-index-work`, not as proven overlap. Small, fast builds can fail this coverage check when their window is too short to observe

## Results and limitations

The output directory contains the dataset identity, before/after query samples, separate baseline and target traffic, the target probe, index observations, database observations, upgrade command output, and a final report. Intermediate artifacts remain when a later check fails

`migration_seconds` comes from newly applied migration-history entries. `upgrade_window_seconds` also includes target readiness and background index completion. Traffic is a small sequential stream, approximately five requests per second per proxy, not a saturation test

Dashboard samples distinguish the first request from repeats but do not claim a cold cache. Full row counts and setup queries already read the database. The default checks reject any request errors, traffic gaps over ten seconds, and repeated-query p95 above twice the baseline with a 250 ms baseline floor

The local integration tests use a small database, one current-version proxy, and a scripted provider. Their query timing budget is deliberately wide to avoid shared-runner noise; deterministic tests verify the default regression threshold separately. They validate the runner, including a real concurrent index build held open by a reader transaction. They do not prove a released-version upgrade, two deployment startup paths, a 1 TB run, partitioned performance, or snapshot restoration
