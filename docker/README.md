# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

> **Just want to run LiteLLM?** `docker-compose.yml` in the repository root runs
> the published image with a Postgres database, the stack the
> [Docker quickstart](https://docs.litellm.ai/docs/proxy/docker_quick_start) documents,
> and it works on its own outside a checkout:
>
> ```bash
> curl -sSLO https://github.com/BerriAI/litellm/raw/main/docker-compose.yml
> printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .env
> docker compose up -d
> ```

## Prerequisites

- Docker
- Docker Compose

## Building and Running the Application

The same `docker-compose.yml` builds from source when you pass `--build`: it builds the `Dockerfile` in the repository root, the one image LiteLLM ships. `docker-entrypoint.sh` next to it is the only entrypoint script; every container starts through it

## One image, many components

Every LiteLLM container runs the same image. The first word of the container command (or the `LITELLM_COMPONENT` environment variable when the command carries only flags) picks the process the container runs, and everything after it is handed to that process unchanged:

| Component    | Runs                                                | Port |
|--------------|-----------------------------------------------------|------|
| `proxy`      | `litellm ...` (everything in one process, default)  | 4000 |
| `gateway`    | `python -m gateway.launch ...` (inference routes)   | 4000 |
| `backend`    | `uvicorn backend.main:app ...` (management routes)  | 4001 |
| `ui`         | nginx serving the static admin UI                   | 3000 |
| `migrations` | `python migrations/run.py`, `prisma migrate deploy` then exit | |
| `metrics`    | `python -m litellm.proxy.prometheus_metrics_server ...` | `--port` |
| `collector`  | `python -m litellm.proxy.collector ...`             | |

```bash
docker run -p 4000:4000 litellm --config /app/config.yaml         # proxy, exactly as before
docker run -p 4000:4000 litellm gateway --port 4000               # componentized data plane
docker run -p 4001:4001 -e LITELLM_COMPONENT=backend litellm      # same, chosen through the env
docker run -p 3000:3000 --read-only --tmpfs /tmp litellm ui       # admin UI behind nginx
docker run -e DATABASE_URL=... litellm migrations                 # one-off schema migration job
docker run -it litellm sh                                         # anything else runs verbatim
```

PgBouncer is not a separate component: `LITELLM_PGBOUNCER_ENABLED=true` starts an in-container PgBouncer in front of `DATABASE_URL` inside `proxy` and `gateway`. `USE_DDTRACE=true` wraps whichever component runs with `ddtrace-run`, and `PROMETHEUS_MULTIPROC_DIR` is emptied of stale samples before any workers fork (the `metrics` and `collector` sidecars only read it, so their restart keeps the live samples)

The image runs as uid `65532` (`nonroot` in the Wolfi base) and also works as an arbitrary uid in gid 0, the shape OpenShift `restricted-v2` assigns, because everything it writes at runtime lives under `/app/.cache`, `/var/lib/litellm` and `/tmp`. Mount those (or set `readOnlyRootFilesystem` with emptyDirs there) for a read-only root filesystem. Prisma's CLI and engines are baked under `/opt/prisma`, so migrations need neither network nor a writable home

### 1. Set the Master and Salt Keys

The proxy signs virtual keys with `LITELLM_MASTER_KEY` and encrypts stored provider credentials with `LITELLM_SALT_KEY`. Compose reads both from a `.env` file in the directory you run it from, so generate them once:

```bash
printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .env
```

Keep the file: regenerating `LITELLM_SALT_KEY` makes credentials already stored in the database unreadable. Provider keys such as `OPENAI_API_KEY` go in the same file, and the whole file is passed to the container

### 2. Build and Run the Containers

```bash
docker compose up -d --build
```

This command builds the image from the root `Dockerfile` and starts the `litellm` and `db` services in detached mode. Without `--build`, `docker compose up` pulls the published `main-stable` image instead. Add `--profile monitoring` to also start Prometheus on port 9090, scraping the proxy with the root `prometheus.yml`

### 3. Verifying the Application is Running

You can check the status of the running containers with the following command:

```bash
docker compose ps
```

To view the logs of the `litellm` container, run:

```bash
docker compose logs -f litellm
```

### 4. Stopping the Application

To stop the running containers, use the following command:

```bash
docker compose down
```

## Hardening

The compose file runs the proxy the way a locked-down cluster would: as the image's non-root user with a read-only root filesystem, every capability dropped, `no-new-privileges` set, and tmpfs mounts only at `/tmp` and `/app/.cache`. The image is built for that, so a change that makes the proxy write anywhere else fails here before it fails in Kubernetes. To try an arbitrary uid in gid 0, the shape OpenShift `restricted-v2` assigns, add `user: "101:0"` to the `litellm` service

Prisma's CLI and engines are baked under `/opt/prisma`, so migrations run without network access. Verify with:

```bash
docker run --rm --network none --entrypoint prisma docker.litellm.ai/berriai/litellm:main-stable --version
```

## Troubleshooting

-   **`required variable LITELLM_MASTER_KEY is missing a value`**: Compose did not find a `.env` file with `LITELLM_MASTER_KEY` and `LITELLM_SALT_KEY` in the directory you ran it from. Generate one as shown above.
-   **`password authentication failed for user "llmproxy"`**: the `litellm_postgres_data` volume was initialised by an older compose file with different credentials. `docker compose down -v` drops it and the next `up` recreates the database.
-   **`build_admin_ui.sh: not found`**: the build context is wrong. Run `docker compose` from the root of the repository.
