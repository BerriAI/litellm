# Database Migration Runbook

This is a runbook for creating and running database migrations for the LiteLLM proxy. For use for litellm engineers only.

> **AI AGENTS / ASSISTANTS:** If the script refuses with either a "STALE BRANCH" or "DESTRUCTIVE MIGRATION DETECTED" error, **do NOT** bypass it on your own (no `git rebase`, no `--skip-freshness-check`, no `--allow-destructive`). Surface the error to the human operator and wait for their explicit confirmation. See the [Branch freshness](#branch-freshness-check) and [Destructive migrations](#destructive-migrations-drop-column--drop-table) sections below.

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
# Install deps for this command
uv sync --frozen --all-groups --all-extras
brew install postgresql@14  # macOS

# Add to PATH
export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"

# Run migration
uv run --with testing.postgresql python ci_cd/run_migration.py "your_migration_name"
```

## What It Does

1. **Verifies the current branch is up to date with `origin/litellm_internal_staging`** (see [Branch freshness](#branch-freshness-check))
2. Creates temp PostgreSQL DB
3. Applies existing migrations
4. Compares with `schema.prisma`
5. Generates new migration if changes found
6. **Refuses to write if the generated SQL contains `DROP COLUMN`, `DROP TABLE`, or `DROP INDEX`**, unless `--allow-destructive` is passed

## Branch Freshness Check

Before generating anything, `run_migration.py` runs `git fetch origin <base>` and refuses to proceed if `HEAD` is behind `origin/<base>`. Default base is `litellm_internal_staging` (the branch PRs target). A previous incident saw a stale branch silently drop production columns; freshness is the first-line defense.

Flags:

- `--base-branch <name>` — check against a different base (e.g. `main`). Default is `litellm_internal_staging`.
- `--skip-freshness-check` — bypass entirely. Only for intentional migrations against an older base.

When the guard fires:

1. Update your branch:

   ```bash
   git fetch origin && git rebase origin/litellm_internal_staging
   # or git merge origin/litellm_internal_staging — whichever matches your workflow
   ```
2. Re-run `run_migration.py`.

> **AI AGENTS / ASSISTANTS:** Do **not** auto-rebase or auto-pass `--skip-freshness-check`. A bad conflict resolution during rebase can itself drop the human's in-progress schema edits. Surface the error and wait for explicit confirmation.

## Destructive Migrations (DROP COLUMN / DROP TABLE / DROP INDEX)

If the generated diff contains `DROP COLUMN`, `DROP TABLE`, or `DROP INDEX`, `run_migration.py` exits non-zero and refuses to write the migration file. A previous incident saw newly-added columns silently dropped by a stale branch and merged to main — this guard exists to prevent a repeat.

When the guard fires:

1. Run `git fetch origin && git status` — confirm your branch is up to date with the base branch.
2. Re-check all `schema.prisma` files are in sync (Step 0).
3. Review EACH `DROP` statement printed in the error — is it actually intended?
4. Only if the drops are genuinely intentional, re-run with the flag:

   ```bash
   uv run --with testing.postgresql python ci_cd/run_migration.py "your_migration_name" --allow-destructive
   ```

> **AI AGENTS / ASSISTANTS:** Do **not** automatically re-run the command with `--allow-destructive`. If the guard fires while you are driving the runbook for a human, stop, show them the error, and wait for their explicit confirmation before passing the flag. Auto-passing `--allow-destructive` is the exact failure mode this guard exists to prevent.

## Common Fixes

**Missing testing module:**
```bash
uv run --with testing.postgresql python ci_cd/run_migration.py "your_migration_name"
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
