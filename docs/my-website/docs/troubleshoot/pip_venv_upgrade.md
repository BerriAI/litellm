# Upgrading LiteLLM Proxy (pip/venv)

Guide for upgrading LiteLLM Proxy when installed via pip in a virtual environment.

:::info Important
Always activate your virtual environment before running any `litellm` or `prisma` commands. All commands in this guide assume you're working inside an activated venv.
:::

## How pip/venv Upgrades Work

There are two pieces that need to stay in sync:

1. **Prisma client** - Generated Python code that talks to the DB
2. **DB schema** - Tables/columns in PostgreSQL

When you upgrade via pip, the `litellm-proxy-extras` package ships with a new `schema.prisma` and a `migrations/` directory. But unlike the Docker image, pip install does NOT automatically regenerate the Prisma client or run migrations. You have to do both manually.

## Upgrade Workflow (pip/venv)

### 1. Stop the proxy

Stop your running LiteLLM proxy instance.

### 2. (Optional) Back up your DB

```bash
pg_dump -h <host> -U <user> -d <db> -F c -f backup_$(date +%Y%m%d).dump
```

### 3. Upgrade the package

```bash
pip install 'litellm[proxy]==<version>'
```

### 4. Regenerate the Prisma client

```bash
prisma generate --schema <venv>/lib/python<version>/site-packages/litellm_proxy_extras/schema.prisma
```

Replace `<venv>` with your virtual environment path and `<version>` with your Python version (e.g., `python3.11`, `python3.12`, `python3.13`).

### 5. Apply DB migrations

You have two options:

**Option A: Just start the proxy** (simplest)

The proxy automatically runs `prisma migrate deploy` on startup, which applies any new migrations.

First, activate your virtual environment:

```bash
source <venv>/bin/activate
```

Then start the proxy:

```bash
litellm --config your_config.yaml --port 4000
```

**Option B: Run manually before starting**

Activate your virtual environment first:

```bash
source <venv>/bin/activate
```

Then run the migration with the explicit schema path:

```bash
prisma migrate deploy --schema <venv>/lib/python<version>/site-packages/litellm_proxy_extras/schema.prisma
```

Replace `<venv>` with your virtual environment path and `<version>` with your Python version (e.g., `python3.11`, `python3.12`, `python3.13`).

### 6. Start the proxy

If you used Option B above, now start the proxy (with venv still activated):

```bash
litellm --config your_config.yaml --port 4000
```

## How to Verify Migrations

### Before applying migrations: Preview what will change

> **Note:** Run `pip install 'litellm[proxy]==<version>'` first (Step 3) so the new `schema.prisma` is available at `<schema-path>`.

```bash
prisma migrate diff \
  --from-url $DATABASE_URL \
  --to-schema-datamodel <schema-path> \
  --script

All migrations should have a `finished_at` timestamp and no `rolled_back_at`.

## Key Things to Know

- **`DISABLE_SCHEMA_UPDATE=true`** env var prevents auto-migration on startup - useful if you want full manual control

- **`prisma db push`** is the nuclear option: force-syncs the DB to match the schema, bypassing migration history. Safe when all changes are additive (new columns/tables), but always have a backup.

- **The `schema.prisma` inside `litellm_proxy_extras` is the source of truth** - always use that one, not one from a different version or from the git repo

## Troubleshooting

If you encounter migration errors, see the [Prisma Migration Troubleshooting Guide](./prisma_migrations).
