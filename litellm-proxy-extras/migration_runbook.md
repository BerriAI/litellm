# Database Migration Runbook

This is a runbook for creating and running database migrations for the LiteLLM proxy. For use for litellm engineers only.

## Quick Start

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

- Update `schema.prisma` first
- Review generated SQL before committing
- Use descriptive migration names
- Never edit existing migration files
- Commit schema + migration together
