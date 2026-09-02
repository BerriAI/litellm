# LiteLLM Hardened Production Stack Documentation

This document outlines the production-grade deployment guidelines for the LiteLLM Proxy infrastructure. The system is engineered as a single, unified architecture offering two configurable egress topologies: **Direct Egress** and **Proxy-Filtered Egress (Squid)**. The underlying design emphasizes container isolation, OWASP-compliant secrets management, and strict network segmentation through a Zero-Trust internal model.

---

## Network Architecture and Traffic Flows

In both execution modes, the architecture enforces strict isolation for sensitive infrastructure components. The PostgreSQL database is placed within a dedicated internal network lacking a default gateway (`backend-internal`), permanently preventing outbound connection attempts to the external internet.

### Topology Option 1: Direct Egress

This configuration optimizes for minimal latency. LiteLLM establishes outbound connections directly to external AI provider endpoints via the host's bridge interface.

```
                  [ Host Firewall / UFW ]
                            │
                     (127.0.0.1:4000)
                            │
┌───────────────────────────┼───────────────────────────────────────────┐
│ DOCKER HOST               ▼                                           │
│                                                                       │
│  [ public-egress Network ]                                            │
│  ┌─────────────────────────────────┐                                  │
│  │         litellm Proxy           │                                  │
│  │    (Reads Secrets via tmpfs)    │                                  │
│  └────────────────┬────────────────┘                                  │
│                   │                                                   │
│  [ backend-internal Network (NO GATEWAY) ]                            │
│  ┌────────────────┴────────────────┐                                  │
│  │           litellm_db            │                                  │
│  │   (PostgreSQL - Isolated)       │                                  │
│  └─────────────────────────────────┘                                  │
└───────────────────┬───────────────────────────────────────────────────┘
                    │ Direct egress via host bridge
                    ▼
          [ Internet / Cloud APIs ]
          ├── api.openai.com
          ├── api.anthropic.com
          └── bedrock.us-east-1.amazonaws.com
```

### Topology Option 2: Proxy-Filtered Egress (Squid)

This configuration introduces outbound traffic control to mitigate data exfiltration risks. All egress requests initiated by LiteLLM are forcibly directed through an internal Squid proxy acting as an egress gateway.

```
                  [ Host Firewall / UFW ]
                            │
                     (127.0.0.1:4000)
                            │
┌───────────────────────────┼───────────────────────────────────────────┐
│ DOCKER HOST               ▼                                           │
│                                                                       │
│  [ public-egress Network ]                                            │
│  ┌─────────────────────────────────┐                                  │
│  │         litellm Proxy           │                                  │
│  │  HTTP_PROXY=http://squid:3128   ├────────┐                         │
│  └────────────────┬────────────────┘        │ HTTP/HTTPS via Proxy    │
│                   │                         ▼                         │
│                   │              ┌────────────────────┐               │
│                   │              │    litellm_squid   │               │
│                   │              │   (Egress Proxy)   │               │
│                   │              └──────────┬─────────┘               │
│  [ backend-internal Network ]               │                         │
│  ┌────────────────┴────────────────┐        │                         │
│  │           litellm_db            │        │ Evaluates Egress Rules  │
│  │   (PostgreSQL - Isolated)       │        │                         │
│  └─────────────────────────────────┘        │                         │
└─────────────────────────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                                  [ Internet / Cloud APIs ]
```

---

## System Prerequisites

- **Operating System:** Linux (Ubuntu LTS or Debian recommended).
- **Access Privileges:** Administrative rights (`root` or `sudo` access) required for systemd unit installation, filesystem permission management, and firewall rule modification.
- **Software Dependencies:**
  - `git` (for repository cloning and sparse-checkout)
  - `docker` (Engine 20.10+)
  - `docker compose` (V2 plugin)
  - `openssl`
  - `envsubst` (provided by `gettext-base`)

---

## Deployment Procedures

### Step 1: Prepare Destination Directory in `/opt/`

Create the production installation path under `/opt/litellm` and navigate into it:

```bash
sudo mkdir -p /opt/litellm
cd /opt/litellm
```

### Step 2: Download Deployment Files via Git Sparse-Checkout

Fetch strictly the target deployment directory without cloning the entire repository history:

```bash
sudo git clone --no-checkout --depth 1 git@github.com:BerriAI/litellm.git src
cd src
sudo git sparse-checkout init --cone
sudo git sparse-checkout set docker-compose
sudo git checkout main
cd docker-compose
```

### Step 3: Validate Provisioned Deployment Files

Verify that the sparse checkout successfully populated the working directory with the required artifacts:

- `deploy.sh`
- `compose.yaml.template`

### Step 4: Set Execution Permissions

Grant execution rights to the deployment script:

```bash
sudo chmod +x deploy.sh
```

### Step 5: Execute Deployment Script

Select the desired topology option using the runtime flags. Append `--install-systemd` if system boot persistence is required:

#### Option 1: Direct Egress Deployment

```bash
# Standard container execution
sudo ./deploy.sh --force-open-firewall

# With Systemd boot persistence
sudo ./deploy.sh --force-open-firewall --install-systemd
```

#### Option 2: Proxy-Filtered Egress Deployment

```bash
# Standard container execution
sudo ./deploy.sh --force-open-firewall --enable-egress-proxy

# With Systemd boot persistence
sudo ./deploy.sh --force-open-firewall --enable-egress-proxy --install-systemd
```

### Automated Execution Lifecycle

Upon invocation, the deployment script executes the following operations:

1. Validates host dependencies and configures system firewall rules (UFW).
2. Generates cryptographically secure secrets in `./secrets/` with restricted `600` permissions mapped to UID/GID `101:101`.
3. Renders the final `compose.yaml` configuration using `compose.yaml.template`.
4. Registers and enables the `litellm.service` systemd unit **only if** `--install-systemd` is passed.
5. Initializes the container stack via `docker compose up -d` (or `systemctl restart litellm.service` when systemd is enabled).

---

## Post-Installation Operations

### Environment Tuning & Operational Flags

To maintain a clean Admin UI and prevent false-positive operational warnings in single-worker deployments, the stack sets specific environment flags:

- **`LITELLM_DISABLE_NO_REDIS_WARNING=true`**: Suppresses the persistent Redis configuration warning banner in the Admin UI. Since this deployment runs as a single, isolated container instance, an in-memory/PostgreSQL setup is sufficient for rate limiting and state persistence without introducing additional Redis infrastructure overhead.
  - _Official Reference:_ [LiteLLM Redis Requirements & Single-Worker Configuration](https://docs.litellm.ai/docs/proxy/redis_requirements)

- **Secrets Ingestion Wrapper (`entrypoint`)**: To comply with Docker Secrets security standards while working around LiteLLM's lack of native `*_FILE` variable parsing, secrets are injected into environment variables at container startup using an inline shell wrapper:
  ```bash
  export DATABASE_URL=$(cat /run/secrets/db_url)
  export LITELLM_MASTER_KEY=$(cat /run/secrets/master_key)
  export LITELLM_SALT_KEY=$(cat /run/secrets/salt_key)
  exec docker/prod_entrypoint.sh
  ```

````

### Service Endpoints

- **Service Base URL:** `http://127.0.0.1:4000`
- **Health Verification Endpoint:** `http://127.0.0.1:4000/health`

### Secrets and Credential Retrieval

Generated credentials are stored in the `./secrets/` directory on the host and are readable exclusively by the `root` user.

#### Retrieve Master API Key (Admin / UI Access)

```bash
sudo cat ./secrets/master_key.txt
````

#### Retrieve Application Salt Key

```bash
sudo cat ./secrets/salt_key.txt
```

#### Retrieve Database Connection String

```bash
sudo cat ./secrets/db_url.txt
```

#### Retrieve PostgreSQL Database Password

```bash
sudo cat ./secrets/db_password.txt
```

---

## Teardown and Cleanup Procedures

To completely remove the container stack, associated systemd units, networks, persistent data volumes, and installation files:

### Step 1: Stop and Remove Systemd Service (If Installed)

```bash
sudo systemctl stop litellm.service 2>/dev/null || true
sudo systemctl disable litellm.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/litellm.service
sudo systemctl daemon-reload
```

### Step 2: Purge Docker Resources and Volumes

Run from `/opt/litellm/src/docker-compose`:

```bash
sudo docker compose down -v --remove-orphans
```

### Step 3: Remove Production Installation Directory

```bash
sudo rm -rf /opt/litellm
```
