# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## Building and Running the Application

To build and run the application, you will use the `docker-compose.yml` file located in the root of the project. This file is configured to use the `Dockerfile.non_root` for a secure, non-root container environment.

### 1. Set the Master Key

The application requires a `MASTER_KEY` for signing and validating tokens. You must set this key as an environment variable before running the application.

Create a `.env` file in the root of the project and add the following line:

```
MASTER_KEY=your-secret-key
```

Replace `your-secret-key` with a strong, randomly generated secret.

### 2. Build and Run the Containers

Once you have set the `MASTER_KEY`, you can build and run the containers using the following command:

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
- Runs the proxy as a non-root user with a read-only rootfs and only two writable tmpfs mounts:
  - `/app/cache` (Prisma/NPM cache; backing `PRISMA_BINARY_CACHE_DIR`, `NPM_CONFIG_CACHE`, `XDG_CACHE_HOME`)
  - `/app/migrations` (Prisma migration workspace; backing `LITELLM_MIGRATION_DIR`)
- Routes all outbound traffic through a local Squid proxy that denies egress, so Prisma migrations must use the cached CLI and engines.

You should also verify offline Prisma behaviour with:

```bash
docker run --rm --network none --entrypoint prisma ghcr.io/berriai/litellm:main-stable --version
```

This command should succeed (showing engine versions) even with `--network none`, confirming that Prisma binaries are available without network access.

## Troubleshooting

-   **`build_admin_ui.sh: not found`**: This error can occur if the Docker build context is not set correctly. Ensure that you are running the `docker-compose` command from the root of the project.
-   **`Master key is not initialized`**: This error means the `MASTER_key` environment variable is not set. Make sure you have created a `.env` file in the project root with the `MASTER_KEY` defined.
