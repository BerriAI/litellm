# Troubleshooting Prisma Migration Errors

Common Prisma migration issues encountered when upgrading or downgrading LiteLLM proxy versions, and how to fix them.

## How Prisma Migrations Work in LiteLLM

- LiteLLM uses [Prisma](https://www.prisma.io/) to manage its PostgreSQL database schema.
- Migration history is tracked in the `_prisma_migrations` table in your database.
- When LiteLLM starts, it runs `prisma migrate deploy` to apply any new migrations.
- Upgrading LiteLLM applies all migrations added since your last applied version.

## Common Errors

### 1. `relation "X" does not exist`

**Example error:**

```
ERROR: relation "LiteLLM_DeletedTeamTable" does not exist
Migration: 20260116142756_update_deleted_keys_teams_table_routing_settings
```

**Cause:** This typically happens after a version rollback. The `_prisma_migrations` table still records migrations from the newer version as "applied," but the underlying database tables were modified, dropped, or never fully created.

**How to fix:**

#### Step 1 — Delete the failed migration entry and restart

Remove the problematic migration from the history so it can be re-applied:

```sql
-- View recent migrations
SELECT migration_name, finished_at, rolled_back_at, logs
FROM "_prisma_migrations"
ORDER BY started_at DESC
LIMIT 10;

-- Delete the failed migration entry
DELETE FROM "_prisma_migrations"
WHERE migration_name = '<failed_migration_name>';
```

After deleting the entry, restart LiteLLM — it will re-apply the migration on startup.

#### Step 2 — If that doesn't work, use `prisma db push`

If deleting the migration entry and restarting doesn't resolve the issue, sync the schema directly:

```bash
DATABASE_URL="<your_database_url>" prisma db push
```

This bypasses migration history and forces the database schema to match the Prisma schema.

---

### 2. `New migrations cannot be applied before the error is recovered from`

**Cause:** A previous migration failed (recorded with an error in `_prisma_migrations`), and Prisma refuses to apply any new migrations until the failure is resolved.

**How to fix:**

1. Find the failed migration:

```sql
SELECT migration_name, finished_at, rolled_back_at, logs
FROM "_prisma_migrations"
WHERE finished_at IS NULL OR rolled_back_at IS NOT NULL
ORDER BY started_at DESC;
```

2. Delete the failed entry and restart LiteLLM:

```sql
DELETE FROM "_prisma_migrations"
WHERE migration_name = '<failed_migration_name>';
```

3. If that doesn't work, use `prisma db push`:

```bash
DATABASE_URL="<your_database_url>" prisma db push
```

---

### 3. Migration state mismatch after version rollback

**Cause:** You upgraded to version X (new migrations applied), rolled back to version Y, then upgraded again. The `_prisma_migrations` table has stale entries for migrations that were partially applied or correspond to a schema state that no longer exists.

**Fix:**

1. Inspect the migration table for problematic entries:

```sql
SELECT migration_name, started_at, finished_at, rolled_back_at, logs
FROM "_prisma_migrations"
ORDER BY started_at DESC
LIMIT 20;
```

2. For each migration that shouldn't be there (i.e., from the version you rolled back from), delete the entry:
     ```sql
     DELETE FROM "_prisma_migrations" WHERE migration_name = '<migration_name>';
     ```

3. Restart LiteLLM to re-run migrations.

4. If that doesn't work, use `prisma db push`:

```bash
DATABASE_URL="<your_database_url>" prisma db push
```
