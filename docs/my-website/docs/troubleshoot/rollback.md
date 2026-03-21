# Safe Rollback Guide

This guide outlines the process for safely rolling back a LiteLLM Proxy deployment to a previous version.

We recommend rolling back to the previous [stable release](https://github.com/BerriAI/litellm/releases). Stable releases come out every week and follow the `main-v<VERSION>-stable` tag convention (e.g., `main-v1.77.2-stable`).

## 1. Determine Rollback Scope

Before proceeding, identify why you are rolling back:
- **Application Logic Error**: Reverting code changes but keeping the database schema.
- **Database Migration Failure**: Reverting changes that included database schema updates.
- **Performance Regression**: Reverting to a known stable version.

## 2. Back Up the Database

> **Always back up before rolling back.** Before making any changes, take a database snapshot or dump. This is your safety net if something goes wrong during the rollback.

```bash
# PostgreSQL example
pg_dump -h <host> -U <user> -d <database> -F c -f litellm_backup_$(date +%Y%m%d_%H%M%S).dump
```

If you are on a managed database (e.g., AWS RDS, GCP Cloud SQL), create a snapshot through your cloud console instead.

## 3. Pre-Rollback Checks

Before reverting, review these items:

- **`LITELLM_SALT_KEY`**: Do **not** change this value during rollback. It is used to encrypt/decrypt your LLM API Key credentials stored in the database. Changing it will make existing credentials unreadable. See [Best Practices for Production](../proxy/prod#8-set-litellm-salt-key).
- **`config.yaml`**: If you added settings specific to the newer version, the older version may not recognize them. Review your config and remove or comment out any settings that were introduced in the version you are rolling back from.
- **`DISABLE_SCHEMA_UPDATE`**: If you use the [Helm PreSync hook for migrations](../proxy/prod#7-use-helm-presync-hook-for-database-migrations-beta) with `DISABLE_SCHEMA_UPDATE=true` on your pods, migrations will **not** auto-run on restart. You will need to handle migration cleanup manually (see Step 5) or re-run the PreSync hook against the older chart version.

## 4. Revert Application Version

Revert your deployment to the previous stable Docker image or Helm chart version.

### Docker
Update your deployment manifest (e.g., K8s Deployment, Docker Compose) to use the previous version:
```yaml
# Example: Reverting to the previous stable release
image: docker.litellm.ai/berriai/litellm:main-v<VERSION>-stable
```

See [all available images](https://github.com/orgs/BerriAI/packages).

### Helm
If you deployed via Helm, use `helm rollback`:
```bash
helm rollback <release-name> [revision-number]
```

## 5. Handle Database Migrations

If you are rolling back to a version that did not have a specific migration, you may need to resolve the migration state in the database.

> LiteLLM uses `prisma migrate deploy` for production (enabled via `USE_PRISMA_MIGRATE=True`). If a migration partially failed or you are reverting code that expects an older schema, you need to clean up the migration history in the `_prisma_migrations` table. See [Best Practices for Production](../proxy/prod#9-use-prisma-migrate-deploy).

### Option A — Delete stale migration entries (recommended)

Connect to your PostgreSQL database and remove migration entries that belong to the version you are rolling back from. This lets LiteLLM re-apply them cleanly if you upgrade again later.

```sql
-- View recent migrations
SELECT migration_name, finished_at, rolled_back_at, logs
FROM "_prisma_migrations"
ORDER BY started_at DESC
LIMIT 10;

-- Delete migration entries from the version you are rolling back from
DELETE FROM "_prisma_migrations"
WHERE migration_name = '<migration_name_from_newer_version>';
```

After deleting the entries, restart LiteLLM — it will re-apply the correct migrations for its version on startup.

> **Note:** If you have `DISABLE_SCHEMA_UPDATE=true` set on your pods, migrations will not auto-run. You need to either temporarily set it to `false`, or re-run the Helm PreSync migration job targeting the older version.

### Option B — Use `prisma migrate resolve` (if you have CLI access)

If you have access to the Prisma CLI (e.g., in a local development environment or a debug container with the `litellm-proxy-extras` package installed):

```bash
DATABASE_URL="<your_database_url>" prisma migrate resolve --rolled-back "<migration_name>"
```

> **Note:** This requires the Prisma CLI to be available in your environment (installed via `prisma-client-py`). If you don't have CLI access (e.g., no shell into the running container), use **Option A** (direct SQL) instead.

### Auto-Recovery Logic
LiteLLM's internal `ProxyExtrasDBManager` automatically attempts to handle idempotent migrations. In many cases, simply rolling back the version and restarting the proxy will be enough if the database changes are additive (e.g., new columns or tables).

## 6. Verification Checklist

After rolling back, verify the health of the system:

- [ ] **Health Endpoint**: Confirm the `/health` endpoint returns `200 OK`.
- [ ] **Check Logs**: Ensure no Prisma errors appear — look for `relation "..." does not exist`, `column "..." does not exist`, or `prisma migrate` failures in the logs.
- [ ] **Spend Tracking**: Run a test completion and confirm the spend is recorded in the `LiteLLM_SpendLogs` table.
- [ ] **Billing (Lago)**: If using Lago for billing (e.g., Lago → Stripe), check proxy logs for `Logged Lago Object` to confirm usage events are being sent.
- [ ] **State Consistency**: If using Redis for caching or rate limiting, consider clearing the cache if the newer version changed the cache key structure.
- [ ] **Admin UI**: Verify the Admin UI loads and shows correct data for keys and teams.

## 7. Troubleshooting

### "New migrations cannot be applied"
If you see this error after a rollback, it means the database has a migration in a "failed" state.
1. Identify the failed migration name (see the SQL query in Step 5).
2. Delete the failed entry from `_prisma_migrations`.
3. Restart the proxy.

### "relation X does not exist"
This typically means a migration entry exists in `_prisma_migrations` but the actual table/column was never created or was dropped.
1. Delete the stale migration entry.
2. Restart LiteLLM so it re-runs the migration.

For more details on Prisma errors, see [Prisma Migrations Troubleshoot](prisma_migrations).
