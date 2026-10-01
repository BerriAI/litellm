# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

> **Just want to run LiteLLM?** This guide builds from source. To run the published
> image instead, use `docker-compose.quickstart.yml` in this directory — the
> two-service stack (gateway + Postgres) that the
> [Docker quickstart](https://docs.litellm.ai/docs/proxy/docker_quick_start) documents:
>
> ```bash
> curl -sSLO https://github.com/BerriAI/litellm/raw/main/docker/docker-compose.quickstart.yml
> printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .env
> docker compose -f docker-compose.quickstart.yml up -d
> ```

## Prerequisites

- Docker
- Docker Compose

## Enable the built-in admin MCP

The images built from `Dockerfile` and `docker/Dockerfile.non_root` include the
[LiteLLM Admin MCP](https://github.com/BerriAI/liteadmin-mcp). Hosting it requires
a valid base LiteLLM Enterprise license. It is disabled by default. Set the
license and enable flag on the existing service, then restart the container:

```yaml
environment:
  LITELLM_LICENSE: "<your-enterprise-license>"
  LITELLM_ENABLE_ADMIN_MCP: "true"
  PROXY_BASE_URL: "https://litellm.example.com"
```

An enabled connector without a valid enterprise license prevents startup with
the standard enterprise-license error. Each MCP request also checks the proxy's
current enterprise status. A base license is sufficient; no additional feature
entitlement is required. The flag being off does not require a license

Keep your existing database, authentication, configuration mount and HTTPS reverse
proxy. The MCP endpoint is `https://litellm.example.com/admin/mcp`, on the same
port as LiteLLM. If the gateway is served under a URL prefix, include that prefix
in the endpoint, for example `https://litellm.example.com/gateway/admin/mcp`

Connect an MCP client that supports Streamable HTTP and bearer headers:

```json
{
  "mcpServers": {
    "litellm-admin": {
      "url": "https://litellm.example.com/admin/mcp",
      "headers": {
        "Authorization": "Bearer <your-personal-proxy-admin-key>"
      }
    }
  }
}
```

Each caller needs a current `proxy_admin` identity, including for read operations.
The connector calls this gateway's management API in process with the caller's
credential. It does not use `LITELLM_BASE_URL` or a shared `LITELLM_API_KEY`

The integration mounts the connector's own ASGI application and uses the standard
`httpx2.ASGITransport` for calls back into the gateway. LiteLLM's outbound HTTP
handler uses `httpx`, while the connector requires `httpx2`. A small ASGI adapter
preserves the original caller's address, scheme and policy headers, which a direct
transport mount would replace with loopback defaults. It retains the complete
gateway middleware and authentication stack; it does not implement HTTP requests
or MCP protocol handling. The adapter explicitly links internal calls to their
parent's admission slot; unrelated requests still acquire their own slots. The
proxy pauses scheduled jobs, drains active requests, closes the connector, and
then cleans up shared resources, including when startup or shutdown fails

`PROXY_BASE_URL` supplies the trusted public origin for Host and Origin validation.
Set `LITELLM_MCP_PUBLIC_URL` to an HTTPS origin if the admin MCP uses a different
public hostname. Without either setting, only the connector's loopback hosts are
accepted. Keep the existing `/mcp` endpoint for the MCP gateway; enabling admin MCP
reserves the `/admin` prefix, including any existing MCP server alias named `admin`

Set `LITELLM_ADMIN_READ_ONLY=true` to disable writes, or
`LITELLM_ADMIN_TOOLS=list_keys,list_teams` to restrict the available operations.
These restrictions apply in addition to the gateway's authorization checks

Embedded deployments return complete results by default, so requests can reach
different workers or replicas. `LITELLM_ADMIN_RESPONSE_VIEW=compact` enables the
connector's paged results. Compact results, including explicit per-call requests
for them, require subsequent `read_admin_result` calls to reach the same worker

The connector source is pinned to a commit and archive checksum in the
`admin-mcp` dependency group and `uv.lock`. Updates ship with the LiteLLM image;
starting the server does not download code. For source development with Python
3.12 or later, install it with `uv sync --extra proxy --group admin-mcp`

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
