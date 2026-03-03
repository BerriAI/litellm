import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Control Plane for Multi-region Architecture (Enterprise)

Learn how to deploy LiteLLM across multiple regions while maintaining centralized administration and avoiding duplication of management overhead.

:::info

✨ This requires LiteLLM Enterprise features.

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/enterprise#trial)

:::

## Overview

When scaling LiteLLM for production use, you may want to deploy multiple instances across different regions or availability zones while maintaining a single point of administration. This guide covers how to set up a distributed LiteLLM deployment with:

- **Regional Worker Instances**: Handle LLM requests for users in specific regions
- **Centralized Admin Instance**: Manages configuration, users, keys, and monitoring

## Architecture Pattern: Regional + Admin Instances

### Typical Deployment Scenario

<Image img={require('../../img/scaling_architecture.png')} />

```mermaid
flowchart TD
    subgraph CP["Control Plane (Admin Instance)"]
        UI["Admin UI"]
        ADMIN_API["Management APIs<br/>/key/*, /user/*, /team/*"]
    end

    subgraph DP_US["Data Plane - US Region"]
        WORKER_US["Worker Instance<br/>/chat/completions, /v1/*"]
    end

    subgraph DP_EU["Data Plane - EU Region"]
        WORKER_EU["Worker Instance<br/>/chat/completions, /v1/*"]
    end

    DB[("Shared PostgreSQL<br/>Database")]

    UI -->|"Manage keys, teams,<br/>models, config"| ADMIN_API
    UI -->|"Switch proxy URL<br/>to view data plane"| WORKER_US
    UI -->|"Switch proxy URL<br/>to view data plane"| WORKER_EU
    ADMIN_API --> DB
    WORKER_US --> DB
    WORKER_EU --> DB

    USER_US["US Users"] -->|"LLM requests"| WORKER_US
    USER_EU["EU Users"] -->|"LLM requests"| WORKER_EU
    ADMIN["Admins"] -->|"Administration"| UI

    style CP fill:#e6f3ff,stroke:#4a90d9
    style DP_US fill:#e6ffe6,stroke:#4a9950
    style DP_EU fill:#e6ffe6,stroke:#4a9950
```

### Benefits of This Architecture

1. **Reduced Management Overhead**: Only one instance needs admin capabilities
2. **Regional Performance**: Users get low-latency access from their region
3. **Centralized Control**: All administration happens from a single interface
4. **Security**: Limit admin access to designated instances only
5. **Cost Efficiency**: Avoid duplicating admin infrastructure

## Configuration

### Admin Instance Configuration

The admin instance handles all management operations and provides the UI.

**Environment Variables for Admin Instance:**
```bash
# Keep admin capabilities enabled (default behavior)
# DISABLE_ADMIN_UI=false          # Admin UI available
# DISABLE_ADMIN_ENDPOINTS=false   # Management APIs available
DISABLE_LLM_API_ENDPOINTS=true      # LLM APIs disabled
DATABASE_URL=postgresql://user:pass@global-db:5432/litellm
LITELLM_MASTER_KEY=your-master-key
```

### Worker Instance Configuration

Worker instances handle LLM requests but have admin capabilities disabled.

**Environment Variables for Worker Instances:**
```bash
# Disable admin capabilities
DISABLE_ADMIN_UI=true           # No admin UI
DISABLE_ADMIN_ENDPOINTS=true    # No management endpoints

DATABASE_URL=postgresql://user:pass@global-db:5432/litellm
LITELLM_MASTER_KEY=your-master-key
```

## Environment Variables Reference

### `DISABLE_ADMIN_UI`

Disables the LiteLLM Admin UI interface.

- **Default**: `false`
- **Worker Instances**: Set to `true`
- **Admin Instance**: Leave as `false` (or don't set)

```bash
# Worker instances
DISABLE_ADMIN_UI=true
```

**Effect**: When enabled, the web UI at `/ui` becomes unavailable.

### `DISABLE_ADMIN_ENDPOINTS`

:::info

✨ This is an Enterprise feature.

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/enterprise#trial)

:::

Disables all management/admin API endpoints.

- **Default**: `false`  
- **Worker Instances**: Set to `true`
- **Admin Instance**: Leave as `false` (or don't set)

```bash
# Worker instances  
DISABLE_ADMIN_ENDPOINTS=true
```

**Disabled Endpoints Include**:
- `/key/*` - Key management
- `/user/*` - User management  
- `/team/*` - Team management
- `/config/*` - Configuration updates
- All other administrative endpoints

**Available Endpoints** (when disabled):
- `/chat/completions` - LLM requests
- `/v1/*` - OpenAI-compatible APIs
- `/vertex_ai/*` - Vertex AI pass-through APIs
- `/bedrock/*` - Bedrock pass-through APIs
- `/health` - Basic health check
- `/metrics` - Prometheus metrics
- All other LLM API endpoints


### `DISABLE_LLM_API_ENDPOINTS`

:::info

✨ This is an Enterprise feature.

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/enterprise#trial)

:::

Disables all LLM API endpoints.

- **Default**: `false`
- **Worker Instances**: Leave as `false` (or don't set)
- **Admin Instance**: Set to `true`

```bash
# Admin instance
DISABLE_LLM_API_ENDPOINTS=true
```


**Disabled Endpoints Include**:
- `/chat/completions` - LLM requests
- `/v1/*` - OpenAI-compatible APIs
- `/vertex_ai/*` - Vertex AI pass-through APIs
- `/bedrock/*` - Bedrock pass-through APIs
- All other LLM API endpoints


**Available Endpoints** (when disabled):
- `/key/*` - Key management
- `/user/*` - User management  
- `/team/*` - Team management
- `/config/*` - Configuration updates
- All other administrative endpoints

### `LITELLM_UI_API_DOC_BASE_URL`

Optional override for the API Reference base URL (used in sample code/docs) when the admin UI runs on a different host than the proxy.


## Usage Patterns

### Client Usage

**For LLM Requests** (use regional endpoints):
```python
import openai

# US users
client_us = openai.OpenAI(
    base_url="https://us.company.com/v1",
    api_key="your-litellm-key"
)

# EU users  
client_eu = openai.OpenAI(
    base_url="https://eu.company.com/v1", 
    api_key="your-litellm-key"
)

response = client_us.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**For Administration** (use admin endpoint):
```python
import requests

# Create a new API key
response = requests.post(
    "https://admin.company.com/key/generate",
    headers={"Authorization": "Bearer sk-1234"},
    json={"duration": "30d"}
)
```

## UI Multi-Proxy Switcher

The LiteLLM Admin UI supports switching between multiple proxy instances directly from the browser. This allows administrators to manage all data planes from a single control plane UI without needing separate browser tabs or logins.

```mermaid
sequenceDiagram
    participant Admin
    participant UI as Admin UI
    participant CP as Control Plane Proxy
    participant DP as Data Plane Proxy

    Admin->>UI: Open Admin UI
    UI->>CP: Load dashboard (default proxy)
    Admin->>UI: Add data plane connection<br/>(URL + API key)
    UI->>DP: Test connection (/health/readiness)
    DP-->>UI: OK (v1.x.x)
    Admin->>UI: Switch to data plane
    UI->>UI: Save selection to localStorage
    UI->>UI: Reload page
    UI->>DP: Load dashboard from data plane
    Admin->>UI: Switch back to control plane
    UI->>CP: Load dashboard from control plane
```

### How It Works

1. Log into the Admin UI on your control plane instance
2. Click the proxy switcher in the navbar (appears after adding a second connection)
3. Select **Manage Connections** to add a data plane
4. Enter the data plane's URL and an API key (master key or virtual key)
5. Click **Test Connection** to verify connectivity
6. Save and switch between proxies using the dropdown

### Adding a Data Plane Connection

From the Admin UI navbar, open **Manage Connections** and provide:

| Field | Description | Example |
|-------|-------------|---------|
| **Name** | A label for this connection | `Production US-East` |
| **Proxy URL** | Full URL of the data plane | `https://us-east.litellm.company.com` |
| **API Key** | Master key or virtual key for auth | `sk-...` |

### Requirements

- **CORS**: Each data plane proxy must have CORS configured to allow requests from the control plane UI's origin. LiteLLM's FastAPI CORS middleware handles this — ensure `allow_origins` includes your control plane domain.
- **API Key**: Since the UI authenticates to remote proxies via API key (not cookie-based SSO), provide a key with sufficient permissions for the operations you need.
- **Network Access**: The admin's browser must be able to reach each data plane URL directly.

### Architecture with Independent Databases

For deployments where each data plane has its own database (true multi-tenancy), the UI switcher allows viewing and managing each independently:

```mermaid
flowchart TD
    subgraph CP["Control Plane"]
        UI["Admin UI<br/>(Proxy Switcher)"]
        CP_PROXY["Admin Proxy"]
        CP_DB[("Control Plane DB")]
    end

    subgraph DP1["Data Plane 1"]
        DP1_PROXY["Worker Proxy"]
        DP1_DB[("Data Plane 1 DB")]
    end

    subgraph DP2["Data Plane 2"]
        DP2_PROXY["Worker Proxy"]
        DP2_DB[("Data Plane 2 DB")]
    end

    UI -->|"Default connection"| CP_PROXY
    UI -.->|"Switch to DP1"| DP1_PROXY
    UI -.->|"Switch to DP2"| DP2_PROXY

    CP_PROXY --> CP_DB
    DP1_PROXY --> DP1_DB
    DP2_PROXY --> DP2_DB

    style CP fill:#e6f3ff,stroke:#4a90d9
    style DP1 fill:#fff3e6,stroke:#d9944a
    style DP2 fill:#fff3e6,stroke:#d9944a
```

Connections are stored in browser `localStorage`, so they persist across sessions and require no backend configuration.

## Related Documentation

- [Virtual Keys](./virtual_keys.md) - Managing API keys and users
- [Health Checks](./health.md) - Monitoring instance health  
- [Prometheus Metrics](./logging.md#prometheus-metrics) - Collecting metrics
- [Production Deployment](./prod.md) - Production best practices 
