# Routing Groups — Agent Instructions

Copy-paste the appropriate section to each agent. Each section is self-contained with all context needed.

---

## AGENT 1: Database & Types (Backend Foundation)

### Instructions to paste:

```
You are implementing the "Routing Groups" feature for LiteLLM. Your job is Agent 1: Database Schema & Pydantic Types.

Read the full plan at ROUTING_GROUPS_PLAN.md for context.

## What is a Routing Group?

A Routing Group is a new first-class entity that lets users configure a named routing pipeline:
- Pick a routing strategy (priority-failover, weighted, cost-based, latency-based, round-robin, etc.)
- Select model deployments (e.g., Nebius, Fireworks AI, Azure)
- Configure priority order or weights
- Assign to teams/keys for access control

## Your Tasks

### 1. Add Prisma Schema

Edit `litellm/proxy/schema.prisma` — add `LiteLLM_RoutingGroupTable` at the end (before any closing comments):

```prisma
model LiteLLM_RoutingGroupTable {
  routing_group_id    String   @id @default(uuid())
  routing_group_name  String   @unique
  description         String?
  routing_strategy    String
  deployments         Json
  fallback_config     Json?    @default("{}")
  retry_config        Json?    @default("{}")
  cooldown_config     Json?    @default("{}")
  settings            Json?    @default("{}")
  assigned_team_ids   String[] @default([])
  assigned_key_ids    String[] @default([])
  is_active           Boolean  @default(true)
  created_at          DateTime @default(now())
  created_by          String
  updated_at          DateTime @default(now()) @updatedAt
  updated_by          String
}
```

Follow the same pattern as `LiteLLM_AccessGroupTable` (in the same file) for style consistency.

### 2. Add Pydantic Types

Edit `litellm/types/router.py` — add these classes near the other config/response types (after `ModelGroupInfo` or near the end):

```python
class RoutingGroupDeployment(BaseModel):
    """A deployment reference within a routing group."""
    model_id: str
    model_name: str
    provider: str
    priority: Optional[int] = None
    weight: Optional[int] = None
    display_name: Optional[str] = None

class RoutingGroupConfig(BaseModel):
    """Full routing group configuration."""
    routing_group_id: Optional[str] = None
    routing_group_name: str
    description: Optional[str] = None
    routing_strategy: str
    deployments: List[RoutingGroupDeployment]
    fallback_config: Optional[dict] = None
    retry_config: Optional[dict] = None
    cooldown_config: Optional[dict] = None
    settings: Optional[dict] = None
    assigned_team_ids: Optional[List[str]] = None
    assigned_key_ids: Optional[List[str]] = None
    is_active: bool = True

class RoutingGroupListResponse(BaseModel):
    routing_groups: List[RoutingGroupConfig]
    total: int
    page: int
    size: int

class RoutingTrace(BaseModel):
    """A single routing decision record for test/simulation."""
    deployment_id: str
    deployment_name: str
    provider: str
    latency_ms: float
    was_fallback: bool
    fallback_depth: int = 0
    status: str  # "success" or "error"
    error_message: Optional[str] = None

class RoutingGroupTestResult(BaseModel):
    """Result of a single test request through a routing group."""
    success: bool
    response_text: Optional[str] = None
    traces: List[RoutingTrace]
    total_latency_ms: float
    final_deployment: str
    final_provider: str

class DeploymentTrafficStats(BaseModel):
    """Per-deployment statistics from a simulation."""
    deployment_id: str
    deployment_name: str
    provider: str
    request_count: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    percent_of_total: float
    priority: Optional[int] = None
    weight: Optional[int] = None

class FlowStep(BaseModel):
    """For priority-failover: represents traffic flow between deployments."""
    from_deployment: Optional[str] = None
    to_deployment: str
    request_count: int
    reason: str  # "primary", "fallback_error", "fallback_rate_limit"

class RoutingGroupSimulationResult(BaseModel):
    """Result of a traffic simulation through a routing group."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    fallback_count: int
    traffic_distribution: List[DeploymentTrafficStats]
    flow_data: Optional[List[FlowStep]] = None

class FailureInjectionConfig(BaseModel):
    """Configuration for injecting failures during simulation."""
    deployment_failure_rates: Dict[str, float]  # {deployment_id: failure_probability 0.0-1.0}
```

### 3. Add Unit Tests

Create `tests/litellm/proxy/test_routing_group_types.py`:
- Test RoutingGroupConfig validation (required fields, optional defaults)
- Test RoutingGroupDeployment with and without priority/weight
- Test RoutingGroupSimulationResult serialization
- Test FailureInjectionConfig validation

### Reference Files (read these first for patterns):
- `litellm/proxy/schema.prisma` — see LiteLLM_AccessGroupTable for table pattern
- `litellm/types/router.py` — see existing BaseModel classes for style
- `tests/litellm/proxy/` — see existing test files for test patterns

### What NOT to do:
- Don't create migration files (Prisma handles this separately)
- Don't modify proxy_server.py or any endpoint files
- Don't add imports inside methods — put them at file top level
```

---

## AGENT 2: Backend API Endpoints (CRUD + Router Integration)

### Instructions to paste:

```
You are implementing the "Routing Groups" feature for LiteLLM. Your job is Agent 2: Backend API Endpoints.

Read the full plan at ROUTING_GROUPS_PLAN.md for context.

**Prerequisite:** Agent 1 has already added the Prisma schema (`LiteLLM_RoutingGroupTable`) and Pydantic types (`RoutingGroupConfig`, `RoutingGroupDeployment`, etc.) in `litellm/types/router.py`.

## What is a Routing Group?

A named, persisted configuration that wraps a list of model deployments + a routing strategy + fallback chain. Under the hood, it creates model groups and fallback configurations that the existing Router consumes.

## Your Tasks

### 1. Create CRUD Endpoints

Create `litellm/proxy/management_endpoints/routing_group_endpoints.py` with these FastAPI endpoints:

```
POST   /v1/routing_group              — Create a routing group
GET    /v1/routing_group              — List routing groups (paginated)
GET    /v1/routing_group/{routing_group_id}  — Get a specific routing group
PUT    /v1/routing_group/{routing_group_id}  — Update a routing group
DELETE /v1/routing_group/{routing_group_id}  — Delete a routing group
POST   /v1/routing_group/{routing_group_id}/test     — Test with single request
POST   /v1/routing_group/{routing_group_id}/simulate — Simulate bulk traffic
```

**Follow the exact pattern** from `litellm/proxy/management_endpoints/access_group_endpoints.py` for:
- Auth checks (use `user_api_key_auth` dependency)
- Prisma DB calls pattern
- Error handling
- Response format

**Create endpoint logic:**
1. Validate routing_strategy is one of: "priority-failover", "simple-shuffle", "least-busy", "latency-based-routing", "cost-based-routing", "usage-based-routing-v2", "weighted"
2. Validate deployments list is non-empty
3. For "priority-failover": validate each deployment has a priority
4. For "weighted": validate each deployment has a weight and weights are reasonable
5. Insert into DB via `prisma_client.db.litellm_routinggrouptable.create()`
6. Call `_sync_routing_group_to_router()` to update the live router
7. Return the created routing group

**Test endpoint (`/test`):**
1. Load the routing group from DB
2. Call `router.acompletion(model=routing_group_name, messages=[...])` with a RoutingTraceCallback registered
3. Return RoutingGroupTestResult with traces showing which deployments were tried

**Simulate endpoint (`/simulate`):**
1. Accept body: `{ num_requests: int, concurrency: int, mock: bool, failure_injection: {...} }`
2. Use the test engine from `litellm/proxy/routing_group_utils/test_engine.py` (Agent 3 builds this)
3. Return RoutingGroupSimulationResult

### 2. Router Integration — `_sync_routing_group_to_router()`

Create a helper function (can be in the same file or a utils file) that translates a RoutingGroupConfig into Router configuration:

**For "priority-failover" strategy:**
- Create separate model groups per priority level:
  - `{routing_group_name}` contains only priority-1 deployments
  - `{routing_group_name}__fallback_p2` contains priority-2 deployments
  - `{routing_group_name}__fallback_p3` contains priority-3 deployments
- Register fallbacks: `[{routing_group_name: [routing_group_name__fallback_p2, routing_group_name__fallback_p3]}]`
- Add deployments to the router via `router.set_model_list()` or `router.add_deployment()`

**For "weighted" strategy:**
- Create a single model group with all deployments, set `weight` in each deployment's `litellm_params`
- Set routing_strategy to "simple-shuffle" (which uses weights when available)

**For "simple-shuffle", "least-busy", "latency-based-routing", "cost-based-routing", "usage-based-routing-v2":**
- Create a single model group with all deployments
- The routing strategy is already set globally on the Router — but document that per-group strategy override is a future enhancement

### 3. Register Endpoints in proxy_server.py

Edit `litellm/proxy/proxy_server.py`:
- Import the router from the new endpoints file
- Add `app.include_router(routing_group_router)` alongside the other management endpoint registrations
- Search for where `access_group_endpoints` is included and add routing_group_endpoints nearby

### 4. Add Startup Sync

In `litellm/proxy/proxy_server.py` in the `ProxyConfig` class:
- In `add_deployment()` or `_update_llm_router()`, after DB models are loaded, also load routing groups from `LiteLLM_RoutingGroupTable`
- Call `_sync_routing_group_to_router()` for each active group

### 5. Unit Tests

Create `tests/litellm/proxy/test_routing_group_endpoints.py`:
- Test CRUD operations (mock the prisma client)
- Test `_sync_routing_group_to_router()` logic for each strategy type
- Test validation (invalid strategy, missing priority for priority-failover, etc.)

### Reference Files (read these first):
- `litellm/proxy/management_endpoints/access_group_endpoints.py` — CRUD pattern to follow
- `litellm/proxy/management_endpoints/model_management_endpoints.py` — model CRUD + router sync pattern
- `litellm/proxy/proxy_server.py` — search for "include_router" and "add_deployment" to find injection points
- `litellm/router.py` — understand `set_model_list()`, `add_deployment()`, fallbacks config
- `litellm/types/router.py` — the types you'll use (RoutingGroupConfig, etc.)

### What NOT to do:
- Don't build the test engine (Agent 3 does that) — just define the endpoint signatures and call into it
- Don't build UI (Agents 4+5)
- Don't modify the Prisma schema (Agent 1 already did)
- Don't add imports inside methods
```

---

## AGENT 3: Test & Simulation Engine (Backend)

### Instructions to paste:

```
You are implementing the "Routing Groups" feature for LiteLLM. Your job is Agent 3: Test & Simulation Engine.

Read the full plan at ROUTING_GROUPS_PLAN.md for context.

**Prerequisite:** Agent 1 has added types (RoutingGroupConfig, RoutingTrace, RoutingGroupTestResult, RoutingGroupSimulationResult, DeploymentTrafficStats, FlowStep, FailureInjectionConfig) in `litellm/types/router.py`. Agent 2 has created the API endpoints that will call into this engine.

## What You're Building

A backend engine that:
1. Sends a single test request through a routing group and captures the full routing trace (which deployment handled it, any fallbacks, latency)
2. Simulates N concurrent requests (with optional mock mode and failure injection) and returns traffic distribution statistics

The UI (Agent 5) will use this data to render an animated flow diagram showing traffic split across deployments.

## Your Tasks

### 1. Create RoutingTraceCallback

Create `litellm/proxy/routing_group_utils/routing_trace_callback.py`:

```python
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RoutingTrace

class RoutingTraceCallback(CustomLogger):
    """
    Temporary callback registered during test/simulation to capture
    which deployments were tried and the outcome of each attempt.
    """

    def __init__(self):
        self.traces: list[RoutingTrace] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Record successful deployment call."""
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})
        trace = RoutingTrace(
            deployment_id=metadata.get("model_id", kwargs.get("model_id", "")),
            deployment_name=kwargs.get("model", ""),
            provider=kwargs.get("custom_llm_provider", litellm_params.get("custom_llm_provider", "")),
            latency_ms=(end_time - start_time).total_seconds() * 1000,
            was_fallback=metadata.get("fallback_depth", 0) > 0,
            fallback_depth=metadata.get("fallback_depth", 0),
            status="success",
        )
        self.traces.append(trace)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Record failed deployment call (before fallback)."""
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})
        exception = kwargs.get("exception", None)
        trace = RoutingTrace(
            deployment_id=metadata.get("model_id", kwargs.get("model_id", "")),
            deployment_name=kwargs.get("model", ""),
            provider=kwargs.get("custom_llm_provider", litellm_params.get("custom_llm_provider", "")),
            latency_ms=(end_time - start_time).total_seconds() * 1000,
            was_fallback=metadata.get("fallback_depth", 0) > 0,
            fallback_depth=metadata.get("fallback_depth", 0),
            status="error",
            error_message=str(exception) if exception else None,
        )
        self.traces.append(trace)
```

Look at existing CustomLogger implementations in `litellm/integrations/` for the exact kwargs structure.

### 2. Create RoutingGroupTestEngine

Create `litellm/proxy/routing_group_utils/test_engine.py`:

```python
class RoutingGroupTestEngine:

    async def test_single_request(
        self,
        routing_group_name: str,
        router: "Router",
        messages: list[dict] | None = None,
        mock: bool = False,
    ) -> RoutingGroupTestResult:
        """
        Send one request through the routing group.
        1. Register a RoutingTraceCallback temporarily
        2. Call router.acompletion(model=routing_group_name, messages=messages)
        3. Collect traces and build RoutingGroupTestResult
        4. Unregister the callback
        """
        ...

    async def simulate_traffic(
        self,
        routing_group_name: str,
        router: "Router",
        routing_group_config: RoutingGroupConfig,
        num_requests: int = 100,
        concurrency: int = 10,
        mock: bool = True,
        failure_injection: FailureInjectionConfig | None = None,
    ) -> RoutingGroupSimulationResult:
        """
        Run N requests through the routing group concurrently.
        1. Register a RoutingTraceCallback temporarily
        2. Run num_requests calls with asyncio.Semaphore(concurrency)
        3. Aggregate results into RoutingGroupSimulationResult
        4. Build flow_data for priority-failover strategy
        5. Unregister the callback
        """
        ...
```

**Key implementation details:**

For `test_single_request`:
- Default messages: `[{"role": "user", "content": "Hello, respond with one word."}]`
- If `mock=True`: Use `litellm.acompletion()` with `mock_response="Mock test response"` parameter (litellm supports this natively)
- Register the trace callback via `litellm.callbacks.append(trace_callback)` before the call and remove it after
- Wrap in try/except to capture both success and failure paths

For `simulate_traffic`:
- Use `asyncio.Semaphore(concurrency)` to limit concurrent requests
- Use `asyncio.gather()` to run all requests
- After all complete, aggregate traces by deployment_id:
  - Count requests, successes, failures per deployment
  - Calculate avg latency per deployment
  - Calculate percent_of_total
- For priority-failover strategy, build `flow_data`:
  - Count how many requests went to each priority level
  - Count how many "fell through" from each level to the next (these are the fallback flows)
  - Build FlowStep objects showing the cascade

For `failure_injection`:
- When mock=True with failure injection, the mock handler should randomly fail based on the configured rates
- One approach: before each mock call, check the deployment's failure rate and if random() < rate, raise a mock error
- This is tricky because the Router picks the deployment internally. Alternative approach:
  - Register a pre-call callback that checks the selected deployment and randomly raises an error
  - Or: use the Router's existing cooldown mechanism to simulate failures

### 3. Create __init__.py

Create `litellm/proxy/routing_group_utils/__init__.py` (empty or with imports).

### 4. Unit Tests

Create `tests/litellm/proxy/test_routing_group_test_engine.py`:
- Test `test_single_request` with a mock router
- Test `simulate_traffic` returns correct aggregation
- Test flow_data is correctly built for priority-failover
- Test failure injection works
- Test the RoutingTraceCallback captures events correctly

### Reference Files (read these first):
- `litellm/integrations/custom_logger.py` — CustomLogger base class, understand async_log_success_event/async_log_failure_event signatures
- `litellm/router.py` — understand acompletion() call, how callbacks are registered
- `litellm/types/router.py` — the types you'll use
- `litellm/router_utils/fallback_event_handlers.py` — understand how fallback depth is tracked

### What NOT to do:
- Don't create API endpoints (Agent 2)
- Don't build UI (Agents 4+5)
- Don't modify the Prisma schema
- Don't add imports inside methods
```

---

## AGENT 4: UI — Routing Group Builder

### Instructions to paste:

```
You are implementing the "Routing Groups" feature for LiteLLM. Your job is Agent 4: UI Routing Group Builder.

Read the full plan at ROUTING_GROUPS_PLAN.md for context.

This is a Next.js (App Router) + React + TypeScript + Ant Design + Tremor + TanStack React Table + TanStack React Query application.

The UI source is at: `ui/litellm-dashboard/`

## What You're Building

A new "Routing Groups" page accessible from the sidebar that lets users:
1. View all routing groups in a table
2. Create a new routing group (pick strategy, select deployments, set priority/weight, assign teams)
3. Edit/delete existing routing groups
4. Navigate to the Live Tester (Agent 5 builds the tester itself)

## Your Tasks

### 1. Add Sidebar Entry

Edit `ui/litellm-dashboard/src/app/(dashboard)/components/Sidebar2.tsx`:
- Add a "Routing Groups" menu item in the sidebar
- Use a suitable icon (e.g., `GitBranchPlus` or `Route` from lucide-react, or `BranchesOutlined` / `NodeIndexOutlined` from @ant-design/icons)
- Place it near "Models & Endpoints" in the menu
- Route: `/routing-groups`

### 2. Create Page Route

Create `ui/litellm-dashboard/src/app/(dashboard)/routing-groups/page.tsx`:

Follow the pattern from `ui/litellm-dashboard/src/app/(dashboard)/teams/page.tsx`:
```tsx
"use client";
import { useAuthorized } from "@/app/(dashboard)/hooks/useAuthorized";
import RoutingGroupsView from "@/components/routing_groups/RoutingGroupsView";

export default function RoutingGroupsPage() {
  const { accessToken, userRole, userId, premiumUser } = useAuthorized();
  return (
    <RoutingGroupsView
      accessToken={accessToken}
      userRole={userRole}
      userId={userId}
    />
  );
}
```

### 3. Create RoutingGroupsView

Create `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsView.tsx`:

Main view with:
- Header: "Routing Groups" title + "Create Routing Group" button
- Table of existing routing groups (RoutingGroupsTable)
- Create/Edit modal (RoutingGroupBuilder)

### 4. Create RoutingGroupsTable

Create `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsTable.tsx`:

Table columns:
| Column | Content |
|--------|---------|
| Name | routing_group_name (clickable link) |
| Strategy | Badge showing strategy type |
| Deployments | Count + mini provider logos |
| Status | Active/Inactive toggle |
| Teams | Number of assigned teams |
| Created | Timestamp |
| Actions | Edit, Delete, Test buttons |

Use Ant Design `Table` or TanStack React Table (see pattern in `ui/litellm-dashboard/src/components/model_dashboard/all_models_table.tsx`).

### 5. Create RoutingGroupBuilder (Create/Edit Modal)

Create `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupBuilder.tsx`:

This is the main form. Use Ant Design `Modal` + `Form` pattern (see `ui/litellm-dashboard/src/app/(dashboard)/teams/components/modals/CreateTeamModal.tsx` for the pattern).

**Form fields:**

1. **Name** — `Form.Item` with `Input` (required)
2. **Description** — `Form.Item` with `Input.TextArea` (optional)
3. **Routing Strategy** — `Form.Item` with `Radio.Group`:
   - "Priority Failover" (value: "priority-failover") — "Ordered. Traffic goes to #1 first, falls back on error"
   - "Weighted Round-Robin" (value: "weighted") — "Split traffic by percentage across providers"
   - "Cost-Based" (value: "cost-based-routing") — "Route to cheapest provider"
   - "Latency-Based" (value: "latency-based-routing") — "Route to fastest provider"
   - "Least Busy" (value: "least-busy") — "Route to provider with fewest in-flight requests"
   - "Usage-Based" (value: "usage-based-routing-v2") — "Route based on TPM/RPM usage"
   - "Round Robin" (value: "simple-shuffle") — "Even distribution across providers"

4. **Deployments** — Custom component (DeploymentSelector):
   - Model search/select dropdown (reuse ModelSelector from `ui/litellm-dashboard/src/components/common_components/ModelSelector.tsx` or build similar)
   - Selected deployments list with drag-to-reorder (for priority-failover)
   - For "priority-failover": show priority number on each item, reorder = reprioritize
   - For "weighted": show weight percentage input/slider on each item
   - For other strategies: simple list, no ordering needed
   - Each item shows: provider logo, model name, remove button

5. **Advanced Settings** — Ant Design `Accordion`:
   - Max Retries (NumberInput)
   - Cooldown Time in seconds (NumberInput)
   - Max Fallbacks (NumberInput)

6. **Access Control**:
   - "Assign to Teams" — multi-select (use pattern from team_dropdown.tsx)
   - "Assign to Keys" — multi-select

**Form submission:**
- Call `routingGroupCreateCall()` or `routingGroupUpdateCall()` from networking.tsx
- Show success notification via NotificationsManager
- Refresh the table

### 6. Create DeploymentSelector

Create `ui/litellm-dashboard/src/components/routing_groups/DeploymentSelector.tsx`:

A compound component:
- Top: Model search dropdown (Ant Design `Select` with search, fetches from `/v2/model/info` or `/model_group/info`)
- Below: List of selected deployments
- For priority-failover: numbered list, drag handles for reorder
- For weighted: each item has a percentage input (Ant Design `Slider` or `InputNumber`)
- Each item: provider icon + name + config controls + remove (X) button

### 7. Create StrategySelector

Create `ui/litellm-dashboard/src/components/routing_groups/StrategySelector.tsx`:

Radio group with strategy options. Each option shows:
- Strategy name (bold)
- Short description
- Optional icon

### 8. Add Networking Functions

Edit `ui/litellm-dashboard/src/components/networking.tsx`:

Add near the end of the file (follow the exact pattern of existing functions like `teamCreateCall`):

```typescript
export const routingGroupCreateCall = async (
  accessToken: string,
  formValues: any,
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group`
    : `/v1/routing_group`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formValues),
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};

export const routingGroupListCall = async (
  accessToken: string,
  page: number = 1,
  size: number = 50,
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group?page=${page}&size=${size}`
    : `/v1/routing_group?page=${page}&size=${size}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};

export const routingGroupGetCall = async (
  accessToken: string,
  routingGroupId: string,
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group/${routingGroupId}`
    : `/v1/routing_group/${routingGroupId}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};

export const routingGroupUpdateCall = async (
  accessToken: string,
  routingGroupId: string,
  formValues: any,
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group/${routingGroupId}`
    : `/v1/routing_group/${routingGroupId}`;
  const response = await fetch(url, {
    method: "PUT",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formValues),
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};

export const routingGroupDeleteCall = async (
  accessToken: string,
  routingGroupId: string,
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group/${routingGroupId}`
    : `/v1/routing_group/${routingGroupId}`;
  const response = await fetch(url, {
    method: "DELETE",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};

export const routingGroupTestCall = async (
  accessToken: string,
  routingGroupId: string,
  messages?: any[],
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group/${routingGroupId}/test`
    : `/v1/routing_group/${routingGroupId}/test`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages }),
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};

export const routingGroupSimulateCall = async (
  accessToken: string,
  routingGroupId: string,
  config: {
    num_requests: number;
    concurrency: number;
    mock: boolean;
    failure_injection?: { deployment_failure_rates: Record<string, number> };
  },
) => {
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/v1/routing_group/${routingGroupId}/simulate`
    : `/v1/routing_group/${routingGroupId}/simulate`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    const errorData = await response.text();
    handleError(errorData);
    throw new Error("Network response was not ok");
  }
  return await response.json();
};
```

### 9. Create React Query Hooks

Create `ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts`:

Follow the pattern from `ui/litellm-dashboard/src/app/(dashboard)/hooks/useModels.ts`:

```typescript
import { useQuery, useMutation } from "@tanstack/react-query";
import { useAuthorized } from "./useAuthorized";
import {
  routingGroupListCall,
  routingGroupGetCall,
  routingGroupCreateCall,
  routingGroupUpdateCall,
  routingGroupDeleteCall,
  routingGroupTestCall,
  routingGroupSimulateCall,
} from "@/components/networking";

export const useRoutingGroups = (page: number = 1, size: number = 50) => {
  const { accessToken } = useAuthorized();
  return useQuery({
    queryKey: ["routingGroups", page, size],
    queryFn: () => routingGroupListCall(accessToken!, page, size),
    enabled: !!accessToken,
  });
};

export const useRoutingGroup = (id: string) => {
  const { accessToken } = useAuthorized();
  return useQuery({
    queryKey: ["routingGroup", id],
    queryFn: () => routingGroupGetCall(accessToken!, id),
    enabled: !!accessToken && !!id,
  });
};
```

### Reference Files (READ THESE FIRST for patterns):
- `ui/litellm-dashboard/src/app/(dashboard)/teams/page.tsx` — page pattern
- `ui/litellm-dashboard/src/app/(dashboard)/teams/components/modals/CreateTeamModal.tsx` — create modal pattern
- `ui/litellm-dashboard/src/app/(dashboard)/components/Sidebar2.tsx` — sidebar pattern
- `ui/litellm-dashboard/src/components/networking.tsx` — API call pattern
- `ui/litellm-dashboard/src/app/(dashboard)/hooks/useModels.ts` — React Query hook pattern
- `ui/litellm-dashboard/src/components/common_components/ModelSelector.tsx` — model selector pattern
- `ui/litellm-dashboard/src/components/common_components/team_dropdown.tsx` — dropdown pattern

### What NOT to do:
- Don't build the Live Tester / Traffic Simulator visualizations (Agent 5 does that)
- Don't build backend endpoints (Agent 2)
- Don't modify Python files
- Do NOT add any new npm dependencies — use only existing libraries (antd, @tremor/react, @tanstack/react-query, @tanstack/react-table, lucide-react, @ant-design/icons)
```

---

## AGENT 5: UI — Live Tester & Traffic Visualization

### Instructions to paste:

```
You are implementing the "Routing Groups" feature for LiteLLM. Your job is Agent 5: Live Tester & Traffic Visualization.

Read the full plan at ROUTING_GROUPS_PLAN.md for the complete context. Your section is "Agent 5" which has detailed specs.

This is a Next.js + React + TypeScript + Ant Design + Tremor + Tailwind CSS application. UI source: `ui/litellm-dashboard/`

## What You're Building

The Live Tester — an animated traffic flow visualization that shows how requests route through a routing group's deployments in real-time. It has TWO visual modes that users can toggle between.

## EXACT DESIGN SPECIFICATION

The component is a dark-themed card that shows an animated flow diagram. There are two modes:

### MODE 1: Ordered Fallback

Visual layout (left to right):
- Left: "client" text label
- An animated flowing teal/cyan stream (thick, with moving gradient animation) going RIGHT
- Center: "LITELLM" label in a circle hub
- From the hub: a thick teal stream flows RIGHT to the first deployment card
- Right side: deployment cards stacked VERTICALLY:
  - Card 1 (top): Teal circle with "N", "Nebius", "Priority 1 · 149ms avg", large "85%" in teal, "285 req"
  - Dashed line going DOWN labeled "fail" between Card 1 and Card 2
  - Card 2: Orange circle with "F", "Fireworks", "Priority 2 · 225ms avg", "10%" in orange, "34 req"
  - Dashed line "fail" between Card 2 and Card 3
  - Card 3: Purple circle with "A", "Azure", "Priority 3 · 321ms avg", "5%" in purple, "17 req"

Header above the card:
- "Live Tester" title + orange badge "Ordered Fallback"
- Subtitle: "Sending to {routing_group_name}"
- Top-right button: "Switch to Weighted"

Stats bar between header and card:
- REQUESTS: 336 | SUCCESS: 100% | AVG LATENCY: 169ms | FALLBACKS: 51

Info box below the card (subtle border):
- "Traffic flows to the primary provider first. When a request fails, it cascades down the priority chain. Thicker streams = more traffic. Dashed red lines show the fallback path."

### MODE 2: Weighted Round-Robin

Visual layout (left to right):
- Left: "client" text label
- Animated stream going RIGHT to "LITELLM" hub circle
- From LITELLM hub: THREE separate colored streams FAN OUT:
  - Thick teal stream CURVES UPWARD to Nebius card (top)
  - Medium orange stream goes STRAIGHT to Fireworks card (middle)
  - Thin purple stream CURVES DOWNWARD to Azure card (bottom)
  - Stream thickness proportional to weight (83% = very thick, 10% = medium, 7% = thin)
  - Each stream uses the provider's color
- Right side: deployment cards spread vertically:
  - Nebius: teal "N", "Weight 83% · 149ms avg", "83%" circle, "278 req"
  - Fireworks: orange "F", "Weight 10% · 225ms avg", "10%" circle, "32 req"
  - Azure: purple "A", "Weight 7% · 321ms avg", "7%" circle, "26 req"

Header: "Live Tester" + teal badge "Weighted Round-Robin"
Top-right: "Switch to Ordered" button
Stats bar: REQUESTS: 336 | SUCCESS: 100% | AVG LATENCY: 169ms | FALLBACKS: 0

Info box: "Traffic is distributed across providers based on weights. Thicker streams = more traffic. Adjust weights in the routing group settings."

## COLOR SCHEME

- Background: dark (#111827 or similar dark gray, NOT pure black)
- Card backgrounds: slightly lighter dark (#1f2937)
- Nebius color: teal (#14b8a6)
- Fireworks color: orange (#f97316)
- Azure color: purple (#8b5cf6)
- Stream animation: flowing gradient or moving dashes
- Text: white (#ffffff) for primary, gray (#9ca3af) for secondary
- Stats labels: gray uppercase, stats values: white bold
- Badge: colored pill (orange for "Ordered Fallback", teal for "Weighted Round-Robin")

## ANIMATED STREAM IMPLEMENTATION

The streams should appear to FLOW from left to right continuously. Implementation approach:

**Using SVG with CSS animation:**

```tsx
// AnimatedStream.tsx
const AnimatedStream = ({ path, color, thickness, animated = true }: Props) => {
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none">
      {/* Background path (dimmer) */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={thickness}
        strokeOpacity={0.15}
      />
      {/* Animated flowing path */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={thickness}
        strokeOpacity={0.6}
        strokeDasharray="12 8"
        className={animated ? "animate-flow" : ""}
      />
      {/* Glow effect */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={thickness + 4}
        strokeOpacity={0.1}
        filter="url(#glow)"
      />
    </svg>
  );
};
```

**CSS animation (add to a style tag or Tailwind config):**
```css
@keyframes flow {
  from { stroke-dashoffset: 20; }
  to { stroke-dashoffset: 0; }
}
.animate-flow {
  animation: flow 0.8s linear infinite;
}
```

**SVG path calculations:**
```
// For ordered fallback (straight line from hub to first card):
// M hubX,hubY L cardX,cardY

// For weighted round-robin (curved streams):
// Top: M hubX,hubY C hubX+80,hubY hubX+80,topCardY cardX,topCardY
// Middle: M hubX,hubY L cardX,midCardY
// Bottom: M hubX,hubY C hubX+80,hubY hubX+80,bottomCardY cardX,bottomCardY
```

## FILES TO CREATE

### 1. `ui/litellm-dashboard/src/components/routing_groups/LiveTester.tsx`

Main container. Props:
```typescript
interface LiveTesterProps {
  routingGroupName: string;
  routingStrategy: string;  // "priority-failover" or "weighted" etc.
  deployments: {
    deployment_id: string;
    provider: string;
    display_name: string;
    priority?: number;
    weight?: number;
    request_count: number;
    success_count: number;
    failure_count: number;
    avg_latency_ms: number;
    percent_of_total: number;
  }[];
  totalRequests: number;
  successRate: number;
  avgLatency: number;
  fallbackCount: number;
  flowSteps?: {
    from_deployment: string | null;
    to_deployment: string;
    request_count: number;
    reason: string;
  }[];
  onSendTest?: () => void;
  onRunSimulation?: (config: SimulationConfig) => void;
  accessToken: string;
  routingGroupId: string;
}
```

Logic:
- Mode toggle state: "ordered" | "weighted"
- Renders StatsBar, then either OrderedFallbackFlow or WeightedRoundRobinFlow
- Renders SimulationControls below
- On "Send Test" or "Run Simulation", calls the API via networking functions, updates state

### 2. `ui/litellm-dashboard/src/components/routing_groups/OrderedFallbackFlow.tsx`

The ordered fallback visualization. Uses SVG for streams + absolutely positioned divs for cards.

Structure:
```
<div className="relative w-full" style={{ height: calculated_height }}>
  <svg className="absolute inset-0 w-full h-full">
    {/* Stream from client to hub */}
    {/* Stream from hub to first card */}
    {/* Dashed fallback lines between cards */}
  </svg>
  {/* Client label (absolute positioned left) */}
  {/* LITELLM hub circle (absolute positioned center-left) */}
  {/* Deployment cards (absolute positioned right, stacked) */}
</div>
```

### 3. `ui/litellm-dashboard/src/components/routing_groups/WeightedRoundRobinFlow.tsx`

The weighted fan-out visualization. SVG curved streams + positioned cards.

### 4. `ui/litellm-dashboard/src/components/routing_groups/DeploymentCard.tsx`

Reusable card component:
```typescript
interface DeploymentCardProps {
  provider: string;          // "nebius", "fireworks_ai", "azure"
  displayName: string;       // "Nebius", "Fireworks", "Azure"
  providerColor: string;     // "#14b8a6", "#f97316", "#8b5cf6"
  label: string;             // "Priority 1" or "Weight 83%"
  avgLatencyMs: number;
  percentage: number;
  requestCount: number;
}
```

Render:
- Dark card with rounded corners and subtle border
- Left: colored circle with first letter of provider
- Center: provider name (bold), label + latency subtitle
- Right: large percentage number (in provider color) + request count

### 5. `ui/litellm-dashboard/src/components/routing_groups/AnimatedStream.tsx`

SVG animated stream component (described above).

### 6. `ui/litellm-dashboard/src/components/routing_groups/StatsBar.tsx`

Horizontal stats row:
```typescript
interface StatsBarProps {
  requests: number;
  successRate: number;
  avgLatency: number;
  fallbacks: number;
}
```

Four columns: label (gray, uppercase, small) + value (white, bold, large).

### 7. `ui/litellm-dashboard/src/components/routing_groups/SimulationControls.tsx`

Below the flow diagram:
- "Run Simulation" button
- Number of requests: Ant Design InputNumber (default 100)
- Mode toggle: Mock / Real (Ant Design Radio.Group)
- Failure injection: per-deployment sliders (Ant Design Slider, 0-100%)
- Loading state during simulation

## PROVIDER COLOR MAPPING

Use this mapping to determine colors:
```typescript
const PROVIDER_COLORS: Record<string, string> = {
  nebius: "#14b8a6",        // teal
  fireworks_ai: "#f97316",  // orange
  azure: "#8b5cf6",         // purple
  openai: "#10b981",        // green
  anthropic: "#d97706",     // amber
  google: "#3b82f6",        // blue
  aws: "#ef4444",           // red
  default: "#6b7280",       // gray
};

const getProviderColor = (provider: string): string => {
  return PROVIDER_COLORS[provider.toLowerCase()] || PROVIDER_COLORS.default;
};
```

## RESPONSIVE DESIGN

- Minimum width: 600px for the flow diagram
- Cards should be at least 200px wide
- On smaller screens, the streams should still be visible (reduce padding)
- The SVG viewBox should scale with the container

## IMPORTANT NOTES

- Do NOT add new npm dependencies. Use only: React, antd, @tremor/react, Tailwind CSS, lucide-react, @ant-design/icons
- Use Tailwind classes for styling where possible
- Use inline SVG for the stream animations (no external SVG library)
- The component should work with dynamic data — number of deployments can vary (2-10)
- Make sure the SVG paths calculate correctly for any number of deployment cards
- The animation should be smooth (use CSS animations, not JS-driven)

## Reference Files (READ FIRST):
- `ui/litellm-dashboard/src/components/networking.tsx` — for routingGroupTestCall, routingGroupSimulateCall
- `ui/litellm-dashboard/src/components/routing_groups/` — Agent 4's files (your components integrate here)
- `ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts` — React Query hooks
- `ui/litellm-dashboard/src/components/common_components/` — reusable components
- `ui/litellm-dashboard/src/components/molecules/notifications_manager.tsx` — for success/error notifications
```

---

## Summary: Agent Assignment Order

| Order | Agent | Can start when | Duration estimate |
|-------|-------|----------------|-------------------|
| 1 | Agent 1 (DB & Types) | Immediately | Small |
| 2a | Agent 2 (API Endpoints) | After Agent 1 | Medium |
| 2b | Agent 4 (UI Builder) | After Agent 1 (parallel with Agent 2) | Medium |
| 3 | Agent 3 (Test Engine) | After Agent 2 | Medium |
| 4 | Agent 5 (Live Tester UI) | After Agents 3 + 4 | Medium-Large |
