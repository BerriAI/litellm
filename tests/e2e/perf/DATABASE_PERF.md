# Database performance dataset

This opt-in tool prepares an unpartitioned, metadata-only spend-log dataset for reusable database snapshots. Migration qualification, dashboard performance tests and partitioned layouts are separate follow-ups. Nothing runs in default E2E collection

## Seed

Apply the selected stable release's migrations to an empty, disposable PostgreSQL database, stop all proxy writers, and set `DATABASE_URL`. Use a fixed UTC midnight before today for the anchor

```sh
uv run python tests/e2e/perf/database_seed.py \
  --baseline-version "$BASELINE_VERSION" \
  --anchor 2026-10-15T00:00:00Z \
  --manifest dataset-manifest.json \
  --confirm-disposable
```

The default target is 1,000,000,000,000 measured bytes in spend logs and their indexes. Daily aggregates and other tables take additional space. For a small validation, add `--target-bytes 100000000 --batch-rows 10000`. Reuse the identical arguments to resume. The seed rejects a different specification, migration history, generator, existing spend logs or a partitioned spend-log table

The synthetic dataset covers 90 days with skewed traffic across keys, teams, users, models, providers and sessions, including failures and cache hits. Prompt and response bodies are empty. Seeded keys and users are blocked and cannot authorize requests

Each batch commits its spend logs, daily aggregates and row checkpoint atomically. A failed batch rolls back all three, and an advisory lock prevents concurrent seed processes. The final manifest records the row count, measured sizes, baseline version, generator and migration hashes, PostgreSQL version, indexes and prompt-storage setting

## Snapshot contract

The checkpoint and manifest live in `litellm_perf_seed.checkpoint` and must be preserved with the database. Stop writers and shut down PostgreSQL cleanly before taking a physical volume snapshot. Record the PostgreSQL major version and manifest alongside the snapshot

Restore each run into a fresh disposable volume using the same PostgreSQL major version. Validate the restored checkpoint, manifest, row count and daily totals before using it. Never attach the saved baseline as a writable test database

Cluster-specific provisioning, snapshot creation and restore commands belong in the private infrastructure repository. This seed alone does not create cloud resources or prove that a 1 TB snapshot has been created
