# Backfill team model ownership

This optional script copies `model_info.team_id` into the new `LiteLLM_ProxyModelTable.team_id` column. It does not rename models, change routing, rewrite metadata, decrypt credentials, or modify timestamps. The schema migrations add the nullable column and its index only; they never run this backfill

Run it with Python 3.10–3.14 in the existing LiteLLM proxy environment, using its existing `psycopg` driver. No additional package dependencies or Prisma client generation are required by the script. Apply the schema migrations first

## Preview and execute

Set `DATABASE_URL` through your existing secret-management mechanism. The script does not accept connection credentials on the command line or print the connection string

```bash
python scripts/backfill_team_model_identity.py
python scripts/backfill_team_model_identity.py --execute
python scripts/backfill_team_model_identity.py
```

The default is a database-enforced read-only dry run. `--dry-run` makes that choice explicit. Every scanned model is reported as one JSON line, identifying its model ID, stored name, current column value, proposed value and action. For example:

```json
{"event":"model","model_id":"example-model-id","model_name":"example-model","status":"pending","from_team_id":null,"to_team_id":"example-team","action":"set_team_id"}
```

`--execute` re-reads the database rather than applying a saved preview. Only rows marked `pending` are eligible. Rows changed since they were read are skipped as `concurrent_change`. Committed rows are reported as `updated`; skipped rows have `action: none`

Global models remain unassigned. Already matching values are unchanged. Conflicting values, malformed metadata, invalid team IDs and references to missing teams are reported and skipped, while valid rows continue. The script never derives ownership from names or chooses between conflicting owners. Legacy metadata stored as a JSON string containing an object is supported

## Live operation and replicas

Dry runs can use read replicas, but their results may lag the primary. `--execute` requires a writable primary; read-only transactions and replicas are rejected before updates. Run the final verification against the primary

PR one retains the existing metadata-based routing behavior. Older proxy versions can still insert unfilled rows or change ownership metadata without updating the new column. Safe reruns fill newly eligible rows but never overwrite conflicts. Complete the proxy rollout, resolve reported discrepancies, then run a full dry run on the primary with `pending: 0` and `issues: 0` before treating the new column as ready for future routing changes. A dry run is an observation, not a guarantee against subsequent model changes

## Batching and resuming

```bash
python scripts/backfill_team_model_identity.py --execute --batch-size 200 --max-batches 10
python scripts/backfill_team_model_identity.py --execute --after-model-id '<cursor from the last committed progress event>'
```

Each bounded batch commits atomically. Lock waits and statements have finite timeouts. An interrupted or failed batch rolls back; earlier committed batches remain valid. Rerunning from the beginning is always supported and is usually simplest. No checkpoint file or extra database table is required

A resumed scan only covers IDs after its cursor. New rows may sort before it, so always finish with a full scan without `--after-model-id`. Each invocation bounds its scan using the greatest ID observed at startup rather than chasing an indefinitely growing table

| Option | Default | Purpose |
| --- | --- | --- |
| `--schema` | URL `schema` parameter, otherwise `current_schema()` | Select the schema explicitly; conflicting schema settings fail |
| `--batch-size` | `500` | Rows per transaction, from 1 to 10,000 |
| `--max-batches` | No limit | Stop after a bounded number of batches |
| `--after-model-id` | Beginning | Resume after a committed cursor |
| `--lock-timeout-ms` | `2000` | Bound lock waits |
| `--statement-timeout-ms` | `30000` | Bound each SQL statement |

PostgreSQL URI and libpq connection strings are accepted. Prisma URL options `schema`, `connection_limit`, `pool_timeout` and `pgbouncer` are handled separately; TLS and other driver options are retained. Ordinary LiteLLM tables with the expected column types and unique IDs are required. Views, partitioned tables, incompatible schema layouts and active row-level filtering are rejected instead of being silently treated as complete scans

Preview requires schema access and SELECT on the model and team tables. Execution additionally requires UPDATE on the model's `team_id` column. No INSERT, DELETE or schema-changing privileges are used by the script

## Results

| Exit code | Meaning |
| --- | --- |
| `0` | Requested scan finished without detected issues; a dry run may still report pending work |
| `1` | Connection, schema, permission or database failure; earlier batches may have committed |
| `2` | Ambiguous or concurrently changed rows were skipped; inspect model events |
| `3` | Stopped at `--max-batches`; resume or rerun |
| `130` | Interrupted; resume or rerun |

Database failures include their SQLSTATE when available. Server exception text and provider parameters are excluded from output. Preserve the last committed progress event if resuming after a failure. The script intentionally offers no overwrite or automatic repair mode

The concurrent index migration is separate from the column migration. If an index build is interrupted, inspect its validity and follow the existing Prisma failed-migration recovery procedure before retrying that migration; do not assume an index with the expected name is valid
