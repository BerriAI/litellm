import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { ControlPlaneArchitecture } from '@site/src/components/ControlPlaneArchitecture';

# [BETA] High Availability Control Plane

Deploy a single LiteLLM UI that manages multiple independent LiteLLM proxy instances, each with its own database, Redis, and master key.

:::info

This is an Enterprise feature.

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Get free 7-day trial key](https://www.litellm.ai/enterprise#trial)

:::

## Why This Architecture?

In the [standard multi-region setup](./control_plane_and_data_plane.md), all instances share a single database and master key. This works, but introduces a shared dependency. If the database goes down, every instance is affected.

The **High Availability Control Plane** takes a different approach:

| | Shared Database (Standard) | High Availability Control Plane |
|---|---|---|
| **Database** | Single shared DB for all instances | Each instance has its own DB |
| **Redis** | Shared Redis | Each instance has its own Redis |
| **Master Key** | Same key across all instances | Each instance has its own key |
| **Failure isolation** | DB outage affects all instances | Failure is isolated to one instance |
| **User management** | Centralized, one user table | Independent, each worker manages its own users |
| **UI** | One UI per admin instance | Single control plane UI manages all workers |

### Benefits

- **True high availability**: no shared infrastructure means no single point of failure
- **Blast radius containment**: a misconfiguration or outage on one worker doesn't affect others
- **Regional isolation**: workers can run in different regions with data residency requirements
- **Simpler operations**: each worker is a self-contained LiteLLM deployment

## Architecture

<ControlPlaneArchitecture />

The **control plane** is a LiteLLM instance that serves the admin UI and knows about all the workers. It is **not a router** — it does not proxy or route any LLM requests. It exists purely so admins can switch between workers and manage them from a single UI.

Each **worker** is a fully independent LiteLLM proxy that handles LLM requests for its region or team. Workers have their own database, Redis, users, keys, teams, and budgets. No infrastructure is shared between workers.

## Setup

### 1. Control Plane Configuration

The control plane needs a `worker_registry` that lists all worker instances.

```yaml title="cp_config.yaml"
model_list: []

general_settings:
  master_key: sk-1234
  database_url: os.environ/DATABASE_URL

worker_registry:
  - worker_id: "worker-a"
    name: "Worker A"
    url: "http://localhost:4001"
  - worker_id: "worker-b"
    name: "Worker B"
    url: "http://localhost:4002"
```

Start the control plane:

```bash
litellm --config cp_config.yaml --port 4000
```

### 2. Worker Configuration

Each worker needs `control_plane_url` in its `general_settings` to enable cross-origin authentication from the control plane UI.

`PROXY_BASE_URL` must also be set for each worker so that SSO callback redirects resolve correctly.

<Tabs>
<TabItem value="worker-a" label="Worker A">

```yaml title="worker_a_config.yaml"
model_list: []

general_settings:
  master_key: sk-worker-a-1234
  database_url: os.environ/WORKER_A_DATABASE_URL
  control_plane_url: "http://localhost:4000"
```

```bash
PROXY_BASE_URL=http://localhost:4001 litellm --config worker_a_config.yaml --port 4001
```

</TabItem>
<TabItem value="worker-b" label="Worker B">

```yaml title="worker_b_config.yaml"
model_list: []

general_settings:
  master_key: sk-worker-b-1234
  database_url: os.environ/WORKER_B_DATABASE_URL
  control_plane_url: "http://localhost:4000"
```

```bash
PROXY_BASE_URL=http://localhost:4002 litellm --config worker_b_config.yaml --port 4002
```

</TabItem>
</Tabs>

:::important
Each worker must have its own `master_key` and `database_url`. The whole point of this architecture is that workers are independent.
:::

### 3. SSO Configuration (Optional)

SSO is configured on the **control plane** instance the same way as a standard LiteLLM proxy. See the [SSO setup guide](./admin_ui_sso.md) for full instructions.

If using SSO, make sure to register each worker URL and the control plane URL as allowed callback URLs in your SSO provider's dashboard.

## How It Works

### Login Flow

1. User visits the control plane UI (`http://localhost:4000/ui`)
2. The login page shows a **worker selector** dropdown listing all registered workers
3. User selects a worker (e.g. "Worker A") and logs in with username/password or SSO
4. The UI authenticates against the **selected worker** using the `/v3/login` endpoint
5. On success, the UI stores the worker's JWT and points all subsequent API calls at the worker
6. The user can now manage keys, teams, models, and budgets on that worker, all from the control plane UI

### Switching Workers

Once logged in, users can switch workers from the **navbar dropdown** without leaving the UI. Switching redirects back to the login page to authenticate against the new worker.

### Discovery

The control plane exposes a `/.well-known/litellm-ui-config` endpoint that the UI reads on load. This endpoint returns:
- `is_control_plane: true`
- The list of workers with their IDs, names, and URLs

This is how the login page knows to show the worker selector.

## Local Testing

To try this out locally, start each instance in a separate terminal:

```bash
# Terminal 1: Control Plane
litellm --config cp_config.yaml --port 4000

# Terminal 2: Worker A
PROXY_BASE_URL=http://localhost:4001 litellm --config worker_a_config.yaml --port 4001

# Terminal 3: Worker B
PROXY_BASE_URL=http://localhost:4002 litellm --config worker_b_config.yaml --port 4002
```

Then open `http://localhost:4000/ui`. You should see the worker selector on the login page.

## Configuration Reference

### Control Plane Settings

| Field | Location | Description |
|---|---|---|
| `worker_registry` | Top-level config | List of worker instances |
| `worker_registry[].worker_id` | Required | Unique identifier for the worker |
| `worker_registry[].name` | Required | Display name shown in the UI |
| `worker_registry[].url` | Required | Full URL of the worker instance |

### Worker Settings

| Field | Location | Description |
|---|---|---|
| `general_settings.control_plane_url` | Required | URL of the control plane instance. Enables `/v3/login` and `/v3/login/exchange` endpoints on this worker. |
| `PROXY_BASE_URL` | Environment variable | The worker's own external URL. Required for SSO callback redirects. |

## Related Documentation

- [Standard Multi-Region Setup](./control_plane_and_data_plane.md) - shared-database architecture for admin/worker split
- [SSO Setup](./admin_ui_sso.md) - configuring SSO for the admin UI
- [Production Deployment](./prod.md) - production best practices
