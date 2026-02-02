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

**Fix Options:**

#### Option A — `prisma db push` (recommended for non-production)

Syncs the Prisma schema directly to the database, bypassing migration history:

```bash
DATABASE_URL="<your_database_url>" prisma db push
```

:::warning
`prisma db push` does not use the migration history. It is best suited for development and QA environments. Use with caution in production.
:::

#### Option B — Clean up `_prisma_migrations` table

Inspect recent migrations and remove the problematic entry so it can be re-applied:

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

After deleting the entry, restart LiteLLM — it will attempt to re-apply the migration.

#### Option C — Mark migration as rolled back

Use the Prisma CLI to mark the migration as rolled back without touching the database:

```bash
prisma migrate resolve --rolled-back <migration_name>
```

This tells Prisma the migration was rolled back, so it will try to re-apply it on the next deploy.

Reference: [Prisma migrate resolve documentation](https://pris.ly/d/migrate-resolve)

---

### 2. `New migrations cannot be applied before the error is recovered from`

**Cause:** A previous migration failed (recorded with an error in `_prisma_migrations`), and Prisma refuses to apply any new migrations until the failure is resolved.

**Fix:** Identify and resolve the blocking migration using [Option A, B, or C](#option-a--prisma-db-push-recommended-for-non-production) above. Then restart LiteLLM.

To find the failed migration:

```sql
SELECT migration_name, finished_at, rolled_back_at, logs
FROM "_prisma_migrations"
WHERE finished_at IS NULL OR rolled_back_at IS NOT NULL
ORDER BY started_at DESC;
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

2. For each migration that shouldn't be there (i.e., from the version you rolled back from), either:
   - **Delete the entry** so it can be cleanly re-applied:
     ```sql
     DELETE FROM "_prisma_migrations" WHERE migration_name = '<migration_name>';
     ```
   - **Mark it as rolled back** via the CLI:
     ```bash
     prisma migrate resolve --rolled-back <migration_name>
     ```

3. Restart LiteLLM to re-run migrations.

For non-production environments, `prisma db push` is the fastest path to a clean state.

---

## General Tips

| Tip | Details |
|-----|---------|
| **Always back up first** | Before any manual migration fix, take a database backup. |
| **`prisma migrate deploy`** | The standard way to apply migrations in production. LiteLLM runs this automatically on startup. |
| **`prisma db push`** | Syncs schema directly — great for dev/QA, use carefully in prod. Does not update `_prisma_migrations`. |
| **`prisma migrate resolve`** | Manually marks a migration as applied or rolled back. Useful for fixing stuck states. |
| **Migration history table** | `_prisma_migrations` in your PostgreSQL database stores all migration state. |
| **Version upgrades** | Prisma applies all migrations added after the last successfully applied one. |

## Further Reading

- [Prisma Migrate: Resolving migration issues](https://pris.ly/d/migrate-resolve)
- [Prisma `db push` documentation](https://www.prisma.io/docs/reference/api-reference/command-reference#db-push)
- [LiteLLM Proxy Documentation](https://docs.litellm.ai/)
