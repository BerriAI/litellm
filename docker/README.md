# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## Quick local UI verification

One command that builds the proxy **from this repo**, with Postgres + Redis, sample config-scoped team budgets, and hot-reload:

```bash
docker compose -f docker-compose.local.yml up --build
```

Then:

- **API + baked UI**: http://localhost:4000/ui/ (sign in as `admin` / `sk-1234`)
- **UI hot-reload**: http://localhost:3000 (Next.js HMR; talks to the proxy on :4000)
- **Teams → Local Demo → Overview** for per-user / per-model budgets, and **My User** for spend progress

Python changes under `./litellm` reload automatically (`--reload` + `PYTHONPATH=/app`). Dashboard source changes are live on :3000. To push a static build into :4000/ui without rebuilding the image, run `npm run build` in `ui/litellm-dashboard` (the `out/` folder is bind-mounted).

Usage page deep links (refresh-safe):

- `/ui/usage?view=my-budgets` — My Budgets view
- `/ui/usage?view=my-budgets&team=local-demo-team` — My Budgets for a team
- `/ui/usage?view=team` — Team Usage
- `/ui/usage?view=global` — Global Usage

Virtual Keys page deep link:

- `/ui/api-keys?virtual_key=<token_hash>` — opens that key's detail view

Navigating away clears those params from the new page; browser back restores the previous URL. Changing the Usage view or selecting another virtual key updates the query string in place.

After first login, add yourself to the demo team (role must be `user`; team admin is enterprise-gated):

```bash
curl -X POST http://localhost:4000/team/member_add \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"team_id":"local-demo-team","member":{"user_id":"default_user_id","role":"user"}}'
```

Then refresh **My User** to see your `$100` / monthly progress bar.

Config used: `docker/local_ui_verify_config.yaml`. First build is slow; later rebuilds reuse cache. Stop with `docker compose -f docker-compose.local.yml down`.

## Building and Running the Application

To build and run the application, you will use the `docker-compose.yml` file located in the root of the project. This file is configured to use the `Dockerfile.non_root` for a secure, non-root container environment.

### 1. Set the Master Key

The application requires a `LITELLM_MASTER_KEY` for signing and validating tokens. You must set this key as an environment variable before running the application.

Create a `.env` file in the root of the project and add the following line:

```
LITELLM_MASTER_KEY=your-secret-key
```

Replace `your-secret-key` with a strong, randomly generated secret.

### 2. Build and Run the Containers

Once you have set the `LITELLM_MASTER_KEY`, you can build and run the containers using the following command:

```bash
docker compose up -d --build
```

This command will:

-   Build the Docker image using `Dockerfile.non_root`.
-   Start the `litellm`, `litellm_db`, and `prometheus` services in detached mode (`-d`).
-   The `--build` flag ensures that the image is rebuilt if there are any changes to the Dockerfile or the application code.

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

## Hardened / Offline Testing

To ensure changes are safe for non-root, read-only root filesystems and restricted egress, always validate with the hardened compose file:

```bash
docker compose -f docker-compose.yml -f docker-compose.hardened.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.hardened.yml up -d
```

This setup:
- Builds from `docker/Dockerfile.non_root` with Prisma engines and Node toolchain baked into the image.
- Runs the proxy as a non-root user with a read-only rootfs and only writable tmpfs mounts:
  - `/app/cache` (Prisma/NPM cache; backing `PRISMA_BINARY_CACHE_DIR`, `NPM_CONFIG_CACHE`, `XDG_CACHE_HOME`)
  - `/app/migrations` (Prisma migration workspace; backing `LITELLM_MIGRATION_DIR`)
- Pre-builds and serves the admin UI from read-only paths:
  - `/var/lib/litellm/ui` (pre-restructured Next.js UI with `.litellm_ui_ready` marker)
  - `/var/lib/litellm/assets` (UI logos and assets)
- Routes all outbound traffic through a local Squid proxy that denies egress, so Prisma migrations must use the cached CLI and engines.

You should also verify offline Prisma behaviour with:

```bash
docker run --rm --network none --entrypoint prisma ghcr.io/berriai/litellm:main-stable --version
```

This command should succeed (showing engine versions) even with `--network none`, confirming that Prisma binaries are available without network access.

## Troubleshooting

-   **`build_admin_ui.sh: not found`**: This error can occur if the Docker build context is not set correctly. Ensure that you are running the `docker-compose` command from the root of the project.
-   **`Master key is not initialized`**: This error means the `LITELLM_MASTER_KEY` environment variable is not set. Make sure you have created a `.env` file in the project root with the `LITELLM_MASTER_KEY` defined.
