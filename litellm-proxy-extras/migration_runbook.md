# Database Migration Runbook

This is a runbook for creating and running database migrations for the LiteLLM proxy. For use for litellm engineers only.

## Step 0: Sync All `schema.prisma` Files

Before doing anything else, make sure all `schema.prisma` files in the repo are in sync. There are multiple copies that must match:

| File | Purpose |
|------|---------|
| `schema.prisma` (repo root) | Source of truth |
| `litellm/proxy/schema.prisma` | Used by the proxy server |
| `litellm-proxy-extras/litellm_proxy_extras/schema.prisma` | Used for migration generation |

**Sync process:**

```bash
# 1. Diff all schema files against the root source of truth
diff schema.prisma litellm/proxy/schema.prisma
diff schema.prisma litellm-proxy-extras/litellm_proxy_extras/schema.prisma

# 2. If there are differences, copy the root schema to all locations
cp schema.prisma litellm/proxy/schema.prisma
cp schema.prisma litellm-proxy-extras/litellm_proxy_extras/schema.prisma

# 3. Verify all files are now identical
diff schema.prisma litellm/proxy/schema.prisma && echo "proxy schema in sync" || echo "MISMATCH"
diff schema.prisma litellm-proxy-extras/litellm_proxy_extras/schema.prisma && echo "extras schema in sync" || echo "MISMATCH"
```

> **Do NOT proceed to migration generation until all schema files are identical.**

## Step 1: Quick Start — Generate Migration

```bash
# Install deps (one time)
pip install testing.postgresql
brew install postgresql@14  # macOS

# Add to PATH
export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"

# Run migration
python ci_cd/run_migration.py "your_migration_name"
```

## What It Does

1. Creates temp PostgreSQL DB
2. Applies existing migrations
3. Compares with `schema.prisma`
4. Generates new migration if changes found

## Common Fixes

**Missing testing module:**
```bash
pip install testing.postgresql
```

**initdb not found:**
```bash
brew install postgresql@14
export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"
```

**Empty migration directory error:**
```bash
rm -rf litellm-proxy-extras/litellm_proxy_extras/migrations/[empty_dir]
```

## Rules

- Sync all `schema.prisma` files first (Step 0)
- Update `schema.prisma` at the repo root first, then sync copies
- Review generated SQL before committing
- Use descriptive migration names
- Never edit existing migration files
- Commit schema + migration together

---

**Done with migration?** See [build_and_publish.md](./build_and_publish.md) to publish a new `litellm-proxy-extras` package.
