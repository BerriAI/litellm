# Routing Groups — Implementation Plan

## Overview

**Routing Groups** are a new first-class entity in LiteLLM that lets users visually compose a routing pipeline: pick a routing strategy, select models (deployments), configure priority/fallback order, and test the group — all from the UI. Once built, a Routing Group can be assigned to teams/keys so developers can call it by name.

### User Story

> "I want to create a Routing Group that first routes to Nebius (low cost), falls back to Fireworks AI on error, then falls back to Azure. I want to test it with a single request to see the flow, and simulate bulk traffic to see the traffic distribution graph. Then I give developers access to it."

---

## Architecture Decisions

### What a Routing Group IS

A Routing Group is a **named, persisted configuration** that wraps:

1. **A list of model deployments** (ordered or unordered depending on strategy)
2. **A routing strategy** (e.g., priority-ordered failover, cost-based, latency-based, round-robin, weighted)
3. **Fallback chain** (derived from the ordered list for priority strategies)
4. **Retry/cooldown settings** (per-group overrides)
5. **Access control** (which teams/keys can use it)

### How it maps to existing Router internals

Under the hood, a Routing Group creates:
- A **model group** (shared `model_name` = the routing group's slug/alias) with all selected deployments
- **`router_settings`** overrides for that group (strategy, fallbacks, retries)
- The existing Router already supports all needed strategies — we just need a UI + persistence layer to configure them as a named group

### What a Routing Group is NOT

- It is NOT a separate Router instance — it's configuration that the single global Router consumes
- It is NOT a replacement for model groups — it's a higher-level abstraction that creates/manages model groups

---

## Critical Design Fixes (Gap Analysis)

These fixes address architectural gaps discovered during codebase analysis. Each is assigned to a specific agent.

### Fix 1: Per-Group Routing Strategy Override (Agent 2)

**Problem:** `Router.routing_strategy` is global — one strategy per Router instance. A "latency-based" routing group and a "cost-based" routing group can't coexist.

**Fix:** Add `model_group_routing_strategy: Dict[str, str]` to the Router (same pattern as the existing `model_group_retry_policy`). In `get_available_deployment()` (router.py:9189), replace `self.routing_strategy` with a lookup:

```python
# In get_available_deployment():
_strategy = self.model_group_routing_strategy.get(model, self.routing_strategy)
# Then use _strategy instead of self.routing_strategy in the if/elif chain
```

During `_sync_routing_group_to_router()`, register the group's strategy:
```python
router.model_group_routing_strategy[routing_group_name] = routing_group_config.routing_strategy
```

This is ~10 lines changed in one method. The pattern already exists for retry policy.

### Fix 2: Config Drift Protection (Agent 2)

**Problem:** `ProxyConfig._update_llm_router()` periodically calls `_delete_deployment()` which removes any model_id NOT in `LiteLLM_ProxyModelTable` + YAML config. Dynamically-created sub-groups (e.g., `prod-model__fallback_p2`) get wiped every refresh cycle.

**Fix:** Two changes in `proxy_server.py`:

1. In `_delete_deployment()`: Before the deletion loop, load routing group managed model IDs and add them to `combined_id_list` so they're protected from deletion:
```python
# After building combined_id_list from DB + config...
routing_group_model_ids = await self._get_routing_group_managed_model_ids()
combined_id_list.extend(routing_group_model_ids)
```

2. In `_update_llm_router()`: After the existing delete/add cycle, re-sync routing groups:
```python
await self._delete_deployment(db_models=models_list)
self._add_deployment(db_models=models_list)
# NEW: Re-sync routing groups from LiteLLM_RoutingGroupTable
await self._sync_all_routing_groups()
```

### Fix 3: Model Name Collision Guard (Agent 2)

**Problem:** A routing group named "gpt-4" would collide with an existing model group called "gpt-4" from regular deployments.

**Fix:** Cross-table uniqueness check on create/update:
- When creating a routing group: query `LiteLLM_ProxyModelTable` to ensure `routing_group_name` doesn't match any existing `model_name`
- Return clear error: `"Name '{name}' is already used by an existing model group. Choose a different name."`
- Also validate on model create side (optional, lower priority)

### Fix 4: Phantom Sub-Group Filtering (Agent 2)

**Problem:** For priority-failover, internal sub-groups like `prod-model__fallback_p2` would show up in `/model/info` and be directly callable.

**Fix:** Tag managed deployments with metadata so they can be filtered:
- Set `model_info.routing_group_managed = True` and `model_info.routing_group_id = "..."` on all sub-group deployments
- In `/model/info` and `/v2/model/info` endpoints, filter out deployments where `routing_group_managed == True`
- In the auth layer, prevent direct calls to sub-groups unless they come through the routing group's fallback chain

### Fix 5: Production Routing Metadata in Spend Logs (Agent 1 types + Agent 2 wiring)

**Problem:** In production, when real traffic flows through a routing group, there's zero visibility. The `RoutingTraceCallback` is only registered during test/simulate.

**Fix:** Add routing group metadata to the standard logging pipeline so it flows to ALL integrations (Langfuse, DataDog, Prometheus, SpendLogs) automatically.

**Agent 1 adds types** — extend `StandardLoggingMetadata` in `litellm/types/utils.py`:
```python
class StandardLoggingMetadata(StandardLoggingUserAPIKeyMetadata):
    # ... existing fields ...
    routing_group_id: Optional[str]       # NEW
    routing_group_name: Optional[str]     # NEW
    routing_strategy: Optional[str]       # NEW
```

**Agent 2 wires it** — in the Router, when a request targets a routing group, inject metadata before the call:
```python
# In _sync_routing_group_to_router or acompletion pre-hook:
if model_group in self.routing_group_configs:
    metadata["routing_group_id"] = config.routing_group_id
    metadata["routing_group_name"] = model_group
    metadata["routing_strategy"] = config.routing_strategy
```

The existing `model_group` field in `StandardLoggingPayload` and `LiteLLM_SpendLogs.model_group` column already captures the routing group name (since routing_group_name IS the model_group). The `LiteLLM_SpendLogs.metadata` JSON field stores the additional routing group fields via `spend_logs_metadata`. No schema migration needed.

**Already existing fields that "just work":**
- `LiteLLM_SpendLogs.model_group` (line 488 of schema.prisma) — auto-populated from Router
- `LiteLLM_SpendLogs.model_id` — which specific deployment handled the request
- `LiteLLM_SpendLogs.metadata` — JSON field with `spend_logs_metadata` dict for custom k/v pairs
- `StandardLoggingPayload.model_group` — flows to all callback integrations

### Fix 6: Stale Deployment Reference Handling (Agent 2)

**Problem:** If a deployment referenced by a routing group gets deleted from the Models page, the routing group has stale `model_id` references.

**Fix:** In the model delete endpoint, check if the model is referenced by any routing group. If so:
- Option A (recommended): Return an error: "This model is used by routing group '{name}'. Remove it from the routing group first."
- Option B: Auto-remove from the routing group and re-sync

### Fix 7: Deployment Health Status in Live Tester (Agent 5)

**Problem:** The Live Tester UI doesn't show if a deployment is currently in cooldown or unhealthy.

**Fix:** Add a `/v1/routing_group/{id}/health` endpoint (Agent 2) that returns per-deployment health:
```python
class DeploymentHealthStatus(BaseModel):
    deployment_id: str
    is_healthy: bool
    in_cooldown: bool
    cooldown_remaining_seconds: Optional[float] = None
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
```

Agent 5 renders this as a green/yellow/red dot on each DeploymentCard.

---

## Agent Breakdown

The work is split into **5 agents** that can work semi-independently. Dependencies are noted.

---

## Agent 1: Database & Types (Backend Foundation)

**Goal:** Define the Routing Group data model, database schema, and Pydantic types.

### Tasks

#### 1.1 Prisma Schema — `LiteLLM_RoutingGroupTable`

Add to `litellm/proxy/schema.prisma`:

```prisma
model LiteLLM_RoutingGroupTable {
  routing_group_id    String   @id @default(uuid())
  routing_group_name  String   @unique          // User-facing name, also the model_group alias
  description         String?
  routing_strategy    String                    // "priority-failover", "simple-shuffle", "least-busy", "latency-based-routing", "cost-based-routing", "usage-based-routing-v2", "weighted"
  deployments         Json                      // Ordered list of deployment refs [{model_id, priority, weight, ...}]
  fallback_config     Json?    @default("{}")   // Custom fallback overrides
  retry_config        Json?    @default("{}")   // Custom retry overrides (num_retries, retry_policy)
  cooldown_config     Json?    @default("{}")   // Custom cooldown overrides
  settings            Json?    @default("{}")   // Additional settings (e.g., max_fallbacks, cooldown_time)
  assigned_team_ids   String[] @default([])     // Teams that can access this group
  assigned_key_ids    String[] @default([])     // Keys that can access this group
  is_active           Boolean  @default(true)
  created_at          DateTime @default(now())
  created_by          String
  updated_at          DateTime @default(now()) @updatedAt
  updated_by          String
}
```

#### 1.2 Pydantic Types — `litellm/types/router.py`

Add types:

```python
class RoutingGroupDeployment(BaseModel):
    """A deployment reference within a routing group."""
    model_id: str                          # References LiteLLM_ProxyModelTable.model_id
    model_name: str                        # The underlying model (e.g., "nebius/meta-llama/...")
    provider: str                          # e.g., "nebius", "fireworks_ai", "azure"
    priority: Optional[int] = None         # For priority-failover: 1 = highest
    weight: Optional[int] = None           # For weighted routing
    display_name: Optional[str] = None     # User-friendly label

class RoutingGroupConfig(BaseModel):
    """Full routing group configuration."""
    routing_group_id: Optional[str] = None
    routing_group_name: str
    description: Optional[str] = None
    routing_strategy: str                  # Enum of supported strategies
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
```

#### 1.3 Extend StandardLoggingMetadata — `litellm/types/utils.py`

Add routing group fields to `StandardLoggingMetadata` (Fix 5 — production observability):

```python
class StandardLoggingMetadata(StandardLoggingUserAPIKeyMetadata):
    # ... existing fields ...
    routing_group_id: Optional[str]       # NEW — UUID of the routing group
    routing_group_name: Optional[str]     # NEW — human-readable name
    routing_strategy: Optional[str]       # NEW — strategy used for this request
    routing_trace: Optional[List[dict]]   # NEW — routing trace: [{deployment_id, model, provider, status, error_class, error_message, latency_ms, priority, weight}]
```

These fields flow automatically to ALL integrations (Langfuse, DataDog, Prometheus, SpendLogs) because `StandardLoggingPayload.metadata` is typed as `StandardLoggingMetadata`. The existing `LiteLLM_SpendLogs.metadata` JSON column stores the full `StandardLoggingMetadata` dict — no schema migration needed. Agent 2 populates `routing_trace` in the fallback handler. Agent 6 renders it in the Logs detail view.

#### 1.4 Migration

Generate Prisma migration for the new table.

### Files to Create/Modify

| File | Action |
|------|--------|
| `litellm/proxy/schema.prisma` | Add `LiteLLM_RoutingGroupTable` |
| `litellm/types/router.py` | Add `RoutingGroupDeployment`, `RoutingGroupConfig`, `RoutingGroupListResponse` |
| `litellm/proxy/db/prisma_client.py` | Ensure new table is accessible |

### Dependencies
- None (foundation layer)

---

## Agent 2: Backend API Endpoints (CRUD + Router Integration)

**Goal:** Build the management API for routing groups and integrate with the existing Router.

### Tasks

#### 2.1 CRUD Endpoints — `litellm/proxy/management_endpoints/routing_group_endpoints.py`

Create a new endpoint module:

```
POST   /v1/routing_group              — Create a routing group
GET    /v1/routing_group              — List routing groups (paginated)
GET    /v1/routing_group/{id}         — Get a specific routing group
PUT    /v1/routing_group/{id}         — Update a routing group
DELETE /v1/routing_group/{id}         — Delete a routing group
POST   /v1/routing_group/{id}/test    — Test a routing group (single request)
POST   /v1/routing_group/{id}/simulate — Simulate traffic through a routing group
```

**Create flow:**
1. Validate strategy + deployments
2. Insert into `LiteLLM_RoutingGroupTable`
3. Call `_sync_routing_group_to_router()` which:
   - Creates a model group with `model_name = routing_group_name`
   - Configures fallbacks based on priority ordering
   - Applies strategy override for the group via `model_group_retry_policy` / router settings

**Test endpoint (`/v1/routing_group/{id}/test`):**
1. Accept optional `messages` param (default: "Hello, respond with one word")
2. Route through the routing group using `router.acompletion(model=routing_group_name)`
3. Return:
   - Response from the LLM
   - Which deployment handled it
   - Latency metrics
   - Any fallbacks triggered (capture via callback)

**Simulate endpoint (`/v1/routing_group/{id}/simulate`):**
1. Accept `num_requests` (e.g., 100), `concurrency` (e.g., 10), `failure_injection` config
2. Run N fake/real requests through the routing group concurrently
3. Collect per-deployment metrics:
   - Request count per deployment
   - Success/failure count
   - Average latency
   - Fallback triggers
4. Return aggregated results for the traffic distribution graph

#### 2.2 Router Integration — `_sync_routing_group_to_router()`

Logic to translate a `RoutingGroupConfig` into Router configuration:

**Implementation approach:**
- For **priority-failover**: Create separate model groups per priority tier, wire them as `fallbacks`. E.g., `routing_group_name` has only priority-1 deployments. `routing_group_name__fallback_p2` has priority-2 deployments. Fallback chain: `[{routing_group_name: [routing_group_name__fallback_p2, routing_group_name__fallback_p3]}]`. Tag all sub-group deployments with `model_info.routing_group_managed = True` (Fix 4).
- For **simple-shuffle / least-busy / latency-based / cost-based / usage-based**: Create a single model group with all deployments. Register per-group strategy via `router.model_group_routing_strategy[name] = strategy` (Fix 1).
- For **weighted**: Create a single model group, set `weight` on each deployment's `litellm_params`. Register strategy as `simple-shuffle` which respects weights.

#### 2.3 Per-Group Routing Strategy Override (Fix 1)

Add `model_group_routing_strategy: Dict[str, str] = {}` to Router, following the existing `model_group_retry_policy` pattern:

1. In `router.py` `__init__`: add `self.model_group_routing_strategy: Dict[str, str] = {}`
2. In `get_available_deployment()` (line 9189) and `async_get_available_deployment()` (line 8845): replace `self.routing_strategy` with:
   ```python
   _strategy = self.model_group_routing_strategy.get(model, self.routing_strategy)
   ```
3. In `update_settings()` (line 8240): add `"model_group_routing_strategy"` to `_allowed_settings`
4. During `_sync_routing_group_to_router()`: register the group's strategy:
   ```python
   router.model_group_routing_strategy[routing_group_name] = config.routing_strategy
   ```

#### 2.4 Config Drift Protection (Fix 2)

Prevent the periodic refresh cycle from deleting routing-group-managed deployments:

1. In `ProxyConfig._delete_deployment()` (proxy_server.py:3485): Before the deletion loop, load routing group managed model IDs and add to `combined_id_list`:
   ```python
   # After building combined_id_list from DB + config...
   routing_group_model_ids = await self._get_routing_group_managed_model_ids()
   combined_id_list.extend(routing_group_model_ids)
   ```

2. In `ProxyConfig._update_llm_router()` (proxy_server.py:3631): After the existing delete/add cycle, re-sync routing groups:
   ```python
   await self._delete_deployment(db_models=models_list)
   self._add_deployment(db_models=models_list)
   # NEW: Re-sync routing groups from LiteLLM_RoutingGroupTable
   await self._sync_all_routing_groups()
   ```

3. Add `_get_routing_group_managed_model_ids()` helper that queries `LiteLLM_RoutingGroupTable` and returns the set of all model IDs managed by routing groups (including sub-group fallback deployments).

#### 2.5 Name Collision Guard (Fix 3)

In the create/update endpoints, before inserting:
```python
# Check that routing_group_name doesn't clash with existing model groups
existing_models = await prisma_client.db.litellm_proxymodeltable.find_many(
    where={"model_name": routing_group_name}
)
if existing_models:
    raise HTTPException(
        status_code=400,
        detail=f"Name '{routing_group_name}' is already used by an existing model group. Choose a different name."
    )
```

#### 2.6 Phantom Sub-Group Filtering (Fix 4)

Tag all routing-group-managed sub-deployments with metadata:
```python
model_info = {
    "routing_group_managed": True,
    "routing_group_id": config.routing_group_id,
    "routing_group_name": config.routing_group_name,
}
```

In `/model/info` and `/v2/model/info` endpoints, filter out models where `model_info.routing_group_managed == True`.

#### 2.7 Production Routing Metadata (Fix 5 — wiring)

In `_sync_routing_group_to_router()`, store routing group configs on the Router instance:
```python
router.routing_group_configs[routing_group_name] = config
```

In the Router's `_acompletion()` or `make_call()`, inject routing group metadata before the call:
```python
if model_group in self.routing_group_configs:
    config = self.routing_group_configs[model_group]
    metadata["routing_group_id"] = config.routing_group_id
    metadata["routing_group_name"] = model_group
    metadata["routing_strategy"] = config.routing_strategy
```

This ensures routing group metadata flows to `StandardLoggingPayload.metadata` → SpendLogs + all integrations automatically.

#### 2.8 Stale Deployment Reference Guard (Fix 6)

In the model delete endpoint (`/model/{model_id}/delete`), before deletion:
```python
# Check if model is referenced by any routing group
routing_groups = await prisma_client.db.litellm_routinggrouptable.find_many()
for rg in routing_groups:
    deployment_ids = [d["model_id"] for d in rg.deployments]
    if model_id in deployment_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete model: it is used by routing group '{rg.routing_group_name}'. Remove it from the routing group first."
        )
```

#### 2.9 Health Endpoint (Fix 7)

Add `GET /v1/routing_group/{id}/health` endpoint that returns per-deployment health:
```python
class DeploymentHealthStatus(BaseModel):
    deployment_id: str
    is_healthy: bool
    in_cooldown: bool
    cooldown_remaining_seconds: Optional[float] = None
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
```

Query the Router's `cooldown_cache` to determine if each deployment is in cooldown.

#### 2.10 Sync on Startup

In `ProxyConfig._update_llm_router()`:
- After loading DB models, also load routing groups from `LiteLLM_RoutingGroupTable`
- Call `_sync_routing_group_to_router()` for each active routing group
- This runs both on initial startup and on every periodic refresh

#### 2.11 Access Control Integration

When a routing group has `assigned_team_ids` / `assigned_key_ids`:
- Automatically add the `routing_group_name` to the team's `models` list (or use access groups)
- On routing group deletion, clean up these references

### Files to Create/Modify

| File | Action |
|------|--------|
| `litellm/proxy/management_endpoints/routing_group_endpoints.py` | **NEW** — CRUD + test + simulate + health |
| `litellm/proxy/proxy_server.py` | Register new router, add sync-on-startup, config drift protection |
| `litellm/proxy/management_endpoints/model_management_endpoints.py` | Add stale reference check on model delete |
| `litellm/router.py` | Add `model_group_routing_strategy` dict + lookup in `get_available_deployment()` + `routing_group_configs` dict |

### Dependencies
- Agent 1 (schema + types must exist)

---

## Agent 3: Test & Simulate Engine (Backend)

**Goal:** Build the test/simulation engine that powers the Live Tester UI.

### Tasks

#### 3.1 Single Request Tester — `litellm/proxy/routing_group_utils/test_engine.py`

```python
class RoutingGroupTestEngine:
    """Executes test requests through a routing group and captures routing metadata."""

    async def test_single_request(
        self,
        routing_group_config: RoutingGroupConfig,
        router: Router,
        messages: Optional[List[dict]] = None,
        mock: bool = False,  # If True, use mock responses (no real LLM calls)
    ) -> RoutingGroupTestResult:
        """
        Send one request through the routing group.
        Capture which deployment handled it, latency, any fallbacks.
        """

    async def simulate_traffic(
        self,
        routing_group_config: RoutingGroupConfig,
        router: Router,
        num_requests: int = 100,
        concurrency: int = 10,
        mock: bool = True,
        failure_injection: Optional[FailureInjectionConfig] = None,
    ) -> RoutingGroupSimulationResult:
        """
        Simulate N requests through the routing group.
        Returns per-deployment traffic distribution and metrics.
        """
```

#### 3.2 Routing Trace Callback — `litellm/proxy/routing_group_utils/routing_trace_callback.py`

A custom `CustomLogger` callback that captures the routing decision path:

```python
class RoutingTraceCallback(CustomLogger):
    """Captures routing decisions for test/simulation reporting."""

    def __init__(self):
        self.traces: List[RoutingTrace] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Record: deployment used, latency, model_group, was_fallback
        trace = RoutingTrace(
            deployment_id=kwargs.get("model_id"),
            deployment_name=kwargs.get("model"),
            provider=kwargs.get("custom_llm_provider"),
            latency_ms=(end_time - start_time).total_seconds() * 1000,
            was_fallback=kwargs.get("litellm_params", {}).get("fallback_depth", 0) > 0,
            status="success",
        )
        self.traces.append(trace)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        # Record failures + which deployment failed
        ...
```

#### 3.3 Failure Injection for Simulation

For simulated traffic, support injecting failures to test failover:

```python
class FailureInjectionConfig(BaseModel):
    deployment_failure_rates: Dict[str, float]  # {model_id: failure_probability}
    # e.g., {"nebius-deploy-1": 0.5} = 50% of requests to Nebius fail
```

When `mock=True` in simulation:
- Don't make real LLM calls
- Use a mock LLM handler that returns instantly (or with configurable latency)
- Apply failure injection to simulate errors
- The Router still executes its full routing logic (strategy selection, fallback, retry)

#### 3.4 Result Types

```python
class RoutingTrace(BaseModel):
    deployment_id: str
    deployment_name: str
    provider: str
    latency_ms: float
    was_fallback: bool
    fallback_depth: int = 0
    status: str  # "success", "error"
    error_message: Optional[str] = None

class RoutingGroupTestResult(BaseModel):
    """Result of a single test request."""
    success: bool
    response_text: Optional[str]
    traces: List[RoutingTrace]  # Full routing path (including failures + fallbacks)
    total_latency_ms: float
    final_deployment: str
    final_provider: str

class RoutingGroupSimulationResult(BaseModel):
    """Result of a traffic simulation."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    traffic_distribution: List[DeploymentTrafficStats]
    # For ordered/priority: flow diagram data
    flow_data: Optional[List[FlowStep]] = None

class DeploymentTrafficStats(BaseModel):
    deployment_id: str
    deployment_name: str
    provider: str
    request_count: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    percent_of_total: float
    was_primary: bool  # True if this was the first-choice deployment

class FlowStep(BaseModel):
    """For priority-failover: shows the flow of traffic."""
    from_deployment: Optional[str]  # None for entry point
    to_deployment: str
    request_count: int
    reason: str  # "primary", "fallback_error", "fallback_rate_limit"
```

### Files to Create

| File | Action |
|------|--------|
| `litellm/proxy/routing_group_utils/__init__.py` | **NEW** |
| `litellm/proxy/routing_group_utils/test_engine.py` | **NEW** — Test + simulation engine |
| `litellm/proxy/routing_group_utils/routing_trace_callback.py` | **NEW** — Routing decision logger |
| `litellm/types/router.py` | Add result types |

### Dependencies
- Agent 1 (types)
- Agent 2 (endpoints call into this engine)

---

## Agent 4: UI — Routing Group Builder

**Goal:** Build the UI page for creating, viewing, editing, and managing routing groups.

### Tasks

#### 4.1 New Sidebar Entry + Route

Add "Routing Groups" to the sidebar (under a "Routing" section or near "Models & Endpoints"):

- **Route:** `/routing-groups`
- **Page file:** `ui/litellm-dashboard/src/app/(dashboard)/routing-groups/page.tsx`

#### 4.2 Page-Level Tabs (Tremor TabGroup)

The routing groups page uses **3 tabs** (following the Tremor `TabGroup` pattern from `TeamsHeaderTabs.tsx` and `ModelsAndEndpointsView.tsx`):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Routing Groups                                      [+ Create] [Refresh]   │
│                                                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                         │
│  │ All Groups   │ │ Health       │ │ Live Tester  │                         │
│  └──────────────┘ └──────────────┘ └──────────────┘                         │
│                                                                              │
│  ╔══════════════════════════════════════════════════════════════════════════╗ │
│  ║  (Tab content renders here)                                             ║ │
│  ╚══════════════════════════════════════════════════════════════════════════╝ │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Tab 1: "All Groups"** — the table of routing groups (existing spec, section 4.3 below)
**Tab 2: "Health"** — health dashboard (section 4.8 below)
**Tab 3: "Live Tester"** — traffic visualization (Agent 5 builds the content)

Implementation (Tremor pattern):
```tsx
<TabGroup>
  <TabList className="flex justify-between mt-2 w-full items-center">
    <div className="flex">
      <Tab>All Groups</Tab>
      <Tab>Health</Tab>
      <Tab>Live Tester</Tab>
    </div>
    <div className="flex items-center space-x-2">
      <Button onClick={openCreateModal}>+ Create Routing Group</Button>
      <Icon icon={RefreshIcon} onClick={onRefresh} />
    </div>
  </TabList>
  <TabPanels>
    <TabPanel><RoutingGroupsTable ... /></TabPanel>
    <TabPanel><RoutingGroupHealthDashboard ... /></TabPanel>
    <TabPanel><LiveTester ... /></TabPanel>
  </TabPanels>
</TabGroup>
```

#### 4.3 Routing Groups List View (Tab 1: "All Groups")

Main page showing all routing groups in a table:

| Column | Description |
|--------|-------------|
| Name | Routing group name (clickable → detail view) |
| Strategy | Routing strategy badge |
| Deployments | Count + provider logos |
| Status | Active/Inactive toggle |
| Teams | Assigned team count |
| Created | Timestamp |
| Actions | Edit, Delete, Test |

#### 4.3 Routing Group Builder (Create/Edit Modal or Page)

**Step-by-step builder flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  Create Routing Group                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Name: [________________________]                            │
│  Description: [________________________]                     │
│                                                              │
│  ── Routing Strategy ──────────────────────────────────────  │
│                                                              │
│  ○ Priority Failover (ordered, first success wins)           │
│  ○ Cost-Based (route to cheapest)                            │
│  ○ Latency-Based (route to fastest)                          │
│  ○ Round Robin (even distribution)                           │
│  ○ Weighted (custom traffic split)                           │
│  ○ Least Busy (fewest in-flight)                             │
│  ○ Usage-Based (TPM/RPM aware)                               │
│                                                              │
│  ── Select Models/Deployments ─────────────────────────────  │
│                                                              │
│  [Search models...                              ] [+ Add]    │
│                                                              │
│  ┌─ Selected Deployments (drag to reorder) ───────────────┐  │
│  │                                                         │  │
│  │  1. ☰ Nebius / meta-llama/Llama-3.3-70B     ⚙️  ✕     │  │
│  │     Priority: 1  |  Provider: nebius                    │  │
│  │                                                         │  │
│  │  2. ☰ Fireworks AI / llama-v3p3-70b          ⚙️  ✕     │  │
│  │     Priority: 2  |  Provider: fireworks_ai              │  │
│  │                                                         │  │
│  │  3. ☰ Azure / gpt-4o                        ⚙️  ✕     │  │
│  │     Priority: 3  |  Provider: azure                     │  │
│  │                                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
│  ── Visual Preview ────────────────────────────────────────  │
│                                                              │
│  For Priority Failover:                                      │
│                                                              │
│  [Request] ──→ [Nebius] ──error──→ [Fireworks] ──error──→ [Azure]  │
│                  │                    │                   │  │
│                  └── ✓ response       └── ✓ response     └── ✓ response  │
│                                                              │
│  For Weighted:                                               │
│                                                              │
│  [Request] ──→ [Nebius 60%] / [Fireworks 30%] / [Azure 10%]  │
│                                                              │
│  ── Advanced Settings (Accordion) ─────────────────────────  │
│                                                              │
│  Max Retries: [3]                                            │
│  Cooldown Time (s): [60]                                     │
│  Max Fallbacks: [5]                                          │
│                                                              │
│  ── Access Control ────────────────────────────────────────  │
│                                                              │
│  Assign to Teams: [Select teams...                  ]        │
│  Assign to Keys:  [Select keys...                   ]        │
│                                                              │
│                               [Cancel]  [Save Routing Group] │
└─────────────────────────────────────────────────────────────┘
```

**UI behavior by strategy:**
- **Priority Failover**: Show numbered list, drag-to-reorder sets priority. Show flow diagram preview.
- **Weighted**: Show slider or percentage input per deployment. Show pie chart preview.
- **Cost/Latency/Least Busy/Usage-Based/Round Robin**: Show unordered list (router auto-selects). Show even-split preview.

#### 4.4 Visual Flow Diagram Component

A lightweight component that renders the routing flow:

For **Priority Failover**:
```
Request ──→ [Nebius] ──error──→ [Fireworks AI] ──error──→ [Azure]
              │                     │                        │
              ✓ return              ✓ return                 ✓ return
```

For **Weighted/Round Robin**:
```
              ┌──→ [Nebius]      60%
Request ──→ ──┼──→ [Fireworks]   30%
              └──→ [Azure]       10%
```

Implementation: Use inline SVG or a simple CSS flexbox layout with arrows (no heavy diagramming library needed). Could use `lucide-react` arrow icons.

#### 4.5 Networking Functions

Add to `networking.tsx`:

```typescript
// CRUD
export const routingGroupCreateCall = async (accessToken: string, data: RoutingGroupConfig) => { ... }
export const routingGroupListCall = async (accessToken: string, page: number, size: number) => { ... }
export const routingGroupGetCall = async (accessToken: string, id: string) => { ... }
export const routingGroupUpdateCall = async (accessToken: string, id: string, data: Partial<RoutingGroupConfig>) => { ... }
export const routingGroupDeleteCall = async (accessToken: string, id: string) => { ... }

// Test & Simulate
export const routingGroupTestCall = async (accessToken: string, id: string, messages?: any[]) => { ... }
export const routingGroupSimulateCall = async (accessToken: string, id: string, config: SimulateConfig) => { ... }
```

#### 4.6 React Query Hooks

Create `ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts`:

```typescript
export const useRoutingGroups = (page, size) => { ... }
export const useRoutingGroup = (id) => { ... }
export const useRoutingGroupTest = () => { ... }  // useMutation
export const useRoutingGroupSimulate = () => { ... }  // useMutation
```

#### 4.7 Health Dashboard Tab — `RoutingGroupHealthDashboard.tsx`

**Goal:** Show the configuration and live health status of all routing groups in one view. Lets users see at a glance: is everything configured correctly? Is anything in cooldown?

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Health                                                                      │
│                                                                              │
│  ┌─ prod-model ──────────────────────────────────────────────────────────┐   │
│  │                                                                        │   │
│  │  Strategy: Priority Failover          Status: Active                   │   │
│  │  Teams: engineering, platform         Keys: 3 assigned                 │   │
│  │                                                                        │   │
│  │  ┌─ Deployments ───────────────────────────────────────────────────┐   │   │
│  │  │                                                                 │   │   │
│  │  │  # Provider      Model                   Priority Health  Avg   │   │   │
│  │  │  ─────────────────────────────────────────────────────────────  │   │   │
│  │  │  1 Nebius        meta-llama/Llama-3.3-70B  P1     ● OK    149ms│   │   │
│  │  │  2 Fireworks AI  llama-v3p3-70b            P2     ● OK    225ms│   │   │
│  │  │  3 Azure         gpt-4o                    P3     ◉ COOL  321ms│   │   │
│  │  │                                                  DOWN 45s      │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                        │   │
│  │  ┌─ Routing Flow ─────────────────────────────────────────────────┐   │   │
│  │  │                                                                 │   │   │
│  │  │  [Request] ──▶ Nebius (P1) ──fail──▶ Fireworks (P2)            │   │   │
│  │  │                                       ──fail──▶ Azure (P3)     │   │   │
│  │  │                                                                 │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                        │   │
│  │  ┌─ Settings ─────────────────────────────────────────────────────┐   │   │
│  │  │  Max Retries: 3  │  Cooldown: 60s  │  Max Fallbacks: 5        │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                        │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─ staging-model ───────────────────────────────────────────────────────┐   │
│  │                                                                        │   │
│  │  Strategy: Weighted Round-Robin       Status: Active                   │   │
│  │  Teams: staging-team                  Keys: 1 assigned                 │   │
│  │                                                                        │   │
│  │  ┌─ Deployments ───────────────────────────────────────────────────┐   │   │
│  │  │                                                                 │   │   │
│  │  │  # Provider      Model                 Weight  Health  Avg     │   │   │
│  │  │  ─────────────────────────────────────────────────────────────  │   │   │
│  │  │  1 OpenAI        gpt-4o                 60%    ● OK    200ms   │   │   │
│  │  │  2 Anthropic     claude-sonnet-4-20250514         40%    ● OK    180ms   │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                        │   │
│  │  ┌─ Routing Flow ─────────────────────────────────────────────────┐   │   │
│  │  │              ┌──▶ OpenAI     60%                                │   │   │
│  │  │  [Request] ──┤                                                  │   │   │
│  │  │              └──▶ Anthropic  40%                                │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                        │   │
│  │  ┌─ Settings ─────────────────────────────────────────────────────┐   │   │
│  │  │  Max Retries: 2  │  Cooldown: 30s  │  Max Fallbacks: 3        │   │   │
│  │  └────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                        │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Each routing group card shows:**

1. **Header row** — Name (bold), Strategy badge, Active/Inactive status tag
2. **Access info** — Assigned teams (comma-separated names), key count
3. **Deployments table** — One row per deployment:
   - `#` (order number)
   - Provider logo + name
   - Model name
   - Priority or Weight (depending on strategy)
   - Health status dot:
     - `● OK` (green) — healthy
     - `◉ COOLDOWN 45s` (yellow) — in cooldown with countdown
     - `● DOWN` (red) — unhealthy
   - Avg latency (from recent requests, via `/health` endpoint)
4. **Routing Flow** — inline simplified flow diagram (reuse `RoutingFlowDiagram` from the builder)
   - For priority-failover: linear chain with `──fail──▶` arrows
   - For weighted: fan-out with percentages
5. **Settings row** — Max retries, Cooldown time, Max fallbacks

**Data source:** Combines `GET /v1/routing_group` (config data) + `GET /v1/routing_group/{id}/health` (live health per deployment). Polls health every 10 seconds.

**Component:** `RoutingGroupHealthDashboard.tsx` — iterates over all routing groups, renders a collapsible card per group (Ant Design `Collapse` or just stacked cards).

### Files to Create/Modify

| File | Action |
|------|--------|
| `ui/litellm-dashboard/src/app/(dashboard)/routing-groups/page.tsx` | **NEW** — Page component |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsView.tsx` | **NEW** — Main view with TabGroup |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsTable.tsx` | **NEW** — List table (Tab 1) |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupBuilder.tsx` | **NEW** — Create/Edit form |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingFlowDiagram.tsx` | **NEW** — Visual flow (shared) |
| `ui/litellm-dashboard/src/components/routing_groups/DeploymentSelector.tsx` | **NEW** — Model picker |
| `ui/litellm-dashboard/src/components/routing_groups/StrategySelector.tsx` | **NEW** — Strategy picker |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupHealthDashboard.tsx` | **NEW** — Health dashboard (Tab 2) |
| `ui/litellm-dashboard/src/components/networking.tsx` | Add CRUD + test/simulate + health calls |
| `ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts` | **NEW** — React Query hooks |
| `ui/litellm-dashboard/src/app/(dashboard)/components/Sidebar2.tsx` | Add "Routing Groups" menu item |

### Dependencies
- Agent 2 (API endpoints must be defined)

---

## Agent 5: UI — Live Tester & Traffic Simulator

**Goal:** Build the Live Tester visualization — the animated traffic flow diagram that shows how requests route through deployments in real-time.

### Design Reference (from mockups)

See `docs/routing_groups_live_tester_*.png` for the exact visual designs.

### Two Modes: Ordered Fallback vs Weighted Round-Robin

The Live Tester has **two visual modes** toggled by a button ("Switch to Weighted" / "Switch to Ordered"):

---

#### MODE 1: Ordered Fallback

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Live Tester   [Ordered Fallback]          [Switch to Weighted ▸]    │
│  Sending to prod-model                                               │
│                                                                      │
│  REQUESTS        SUCCESS        AVG LATENCY        FALLBACKS         │
│  336             100%           169ms               51                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                                                                │   │
│  │                                                                │   │
│  │  client ═══════════════▶ (LITELLM) ═══════════▶ ┌───────────┐ │   │
│  │          animated teal                          │  N         │ │   │
│  │          stream                                 │  Nebius    │ │   │
│  │                                                 │  Priority 1│ │   │
│  │                                                 │  149ms avg │ │   │
│  │                                                 │  85%  285rq│ │   │
│  │                                                 └─────┬──────┘ │   │
│  │                                                   fail│(dashed)│   │
│  │                                                       ▼        │   │
│  │                                                 ┌───────────┐  │   │
│  │                                                 │  F         │  │   │
│  │                                                 │  Fireworks │  │   │
│  │                                                 │  Priority 2│  │   │
│  │                                                 │  225ms avg │  │   │
│  │                                                 │  10%  34rq │  │   │
│  │                                                 └─────┬──────┘  │   │
│  │                                                   fail│(dashed) │   │
│  │                                                       ▼         │   │
│  │                                                 ┌───────────┐   │   │
│  │                                                 │  A         │   │   │
│  │                                                 │  Azure     │   │   │
│  │                                                 │  Priority 3│   │   │
│  │                                                 │  321ms avg │   │   │
│  │                                                 │  5%   17rq │   │   │
│  │                                                 └────────────┘   │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ℹ️ Traffic flows to the primary provider first. When a request      │
│     fails, it cascades down the priority chain. Thicker streams =    │
│     more traffic. Dashed red lines show the fallback path.           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key visual elements for Ordered Fallback:**
- Left side: "client" label with an **animated teal/cyan stream** (thick, flowing particles/gradient) going right
- Center: **LITELLM circle** hub — the stream flows into it
- From LITELLM hub: a **single thick teal stream** flows right to the first (primary) deployment card
- **Deployment cards** stacked vertically on the right side, each card contains:
  - Colored letter icon (N=teal, F=orange, A=purple) in a circle
  - Provider name (bold)
  - "Priority N · Xms avg" subtitle
  - Large percentage circle (colored matching the provider)
  - Request count ("285 req")
- Between cards: **dashed lines** labeled "fail" showing the fallback cascade path
- Stream thickness is proportional to traffic volume
- The teal stream is thickest going to the primary (Priority 1) deployment
- Color scheme: dark background (#1a1a2e or similar dark card), teal/cyan (#14b8a6) for primary stream, provider-specific colors for each card

---

#### MODE 2: Weighted Round-Robin

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Live Tester   [Weighted Round-Robin]       [Switch to Ordered ▸]    │
│  Sending to prod-model                                               │
│                                                                      │
│  REQUESTS        SUCCESS        AVG LATENCY        FALLBACKS         │
│  336             100%           169ms               0                 │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                                                                │   │
│  │                                        ┌───────────┐           │   │
│  │                          ╔═════════════▶│  N  Nebius │           │   │
│  │                          ║  thick teal  │  Wt 83%   │           │   │
│  │                          ║              │  149ms avg │           │   │
│  │  client ═══════▶ (LITELLM)              │  83% 278rq│           │   │
│  │                          ║              └───────────┘           │   │
│  │                          ║              ┌────────────┐          │   │
│  │                          ╠═════════════▶│  F Fireworks│          │   │
│  │                          ║  med orange  │  Wt 10%    │          │   │
│  │                          ║              │  225ms avg  │          │   │
│  │                          ║              │  10%  32rq  │          │   │
│  │                          ║              └────────────┘          │   │
│  │                          ║              ┌───────────┐           │   │
│  │                          ╚═════════════▶│  A  Azure  │           │   │
│  │                             thin purple │  Wt 7%    │           │   │
│  │                                         │  321ms avg │           │   │
│  │                                         │  7%   26rq│           │   │
│  │                                         └───────────┘           │   │
│  │                                                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ℹ️ Traffic is distributed across providers based on weights.         │
│     Thicker streams = more traffic. Adjust weights in routing        │
│     group settings.                                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key visual elements for Weighted Round-Robin:**
- Left side: Same "client" label with animated stream → LITELLM hub
- From LITELLM hub: **THREE separate colored streams** fan out to the right, each curving to its destination:
  - **Top stream** (teal, thick): curves UP to Nebius card — thickness = 83%
  - **Middle stream** (orange, medium): goes STRAIGHT to Fireworks card — thickness = 10%
  - **Bottom stream** (purple, thin): curves DOWN to Azure card — thickness = 7%
- Stream thickness is proportional to the weight percentage
- Each stream uses the provider's color
- **Deployment cards** on the right side, each contains:
  - Colored letter icon
  - Provider name
  - "Weight N% · Xms avg" subtitle
  - Large percentage circle
  - Request count
- NO dashed fallback lines (weighted mode doesn't have fallback cascade)
- The streams should have a **smooth curved path** (CSS bezier or SVG path) — NOT straight lines

---

### Tasks

#### 5.1 Live Tester Component — `LiveTester.tsx`

The main container component with:

**Header section:**
- Title: "Live Tester" + strategy badge (orange for "Ordered Fallback", teal for "Weighted Round-Robin")
- Subtitle: "Sending to {routing_group_name}"
- Toggle button: "Switch to Weighted" / "Switch to Ordered"

**Stats bar (horizontal row of 4 metrics):**
- REQUESTS: total count
- SUCCESS: percentage
- AVG LATENCY: in ms
- FALLBACKS: count (0 for weighted mode)

**Flow visualization area** (dark background card):
- Renders either `OrderedFallbackFlow` or `WeightedRoundRobinFlow` based on mode

**Info box** at bottom (subtle border, light text):
- Contextual explanation of the current mode

**Controls:**
- "Send Test Request" button — sends one real request, updates flow with result
- "Send Mock Request" button — simulates one request
- "Run Simulation (N requests)" — bulk test, updates all stats

#### 5.2 Ordered Fallback Flow Component — `OrderedFallbackFlow.tsx`

Renders the left-to-right flow with vertical fallback cascade:

**Implementation using SVG + CSS:**
1. **Client label** (left, vertically centered)
2. **Animated stream** — SVG `<path>` with CSS animation (moving gradient or dashed animation) from client to LITELLM hub
3. **LITELLM circle** — centered hub node
4. **Primary stream** — SVG path from hub to first deployment card (thickness based on traffic %)
5. **Deployment cards** — absolutely positioned divs on the right side, stacked vertically
6. **Fallback dashed lines** — SVG dashed paths between deployment cards, labeled "fail"

**Animated stream effect:**
- Use CSS `@keyframes` with `stroke-dashoffset` animation on SVG paths
- Or use a gradient that shifts position over time
- The stream should appear to "flow" from left to right continuously

**Each deployment card contains:**
- Provider icon circle (colored: teal=#14b8a6, orange=#f97316, purple=#8b5cf6)
- Provider name (bold, white text)
- "Priority {N} · {X}ms avg" (gray subtext)
- Percentage circle (large, colored)
- Request count label

#### 5.3 Weighted Round-Robin Flow Component — `WeightedRoundRobinFlow.tsx`

Renders the fan-out flow with curved colored streams:

**Implementation using SVG + CSS:**
1. Same **Client label** and **LITELLM hub** on the left
2. **Multiple curved streams** from hub to each deployment:
   - Use SVG `<path>` with cubic bezier curves (`C` command)
   - Top stream curves upward, middle goes straight, bottom curves downward
   - Each stream has the provider's color
   - Stream `stroke-width` is proportional to weight percentage (e.g., 83% = 12px, 10% = 4px, 7% = 2px)
   - Same flowing animation as ordered mode
3. **Deployment cards** spread vertically on the right side

**Stream path calculation:**
```
// Hub position: (centerX, centerY)
// Card positions: (rightX, topY), (rightX, midY), (rightX, bottomY)
// Use cubic bezier: M hubX,hubY C controlX1,controlY1 controlX2,controlY2 cardX,cardY
```

#### 5.4 Shared Sub-Components

**`DeploymentCard.tsx`** — Reusable card for each deployment:
- Props: `provider`, `providerColor`, `label` (e.g., "Priority 1" or "Weight 83%"), `avgLatencyMs`, `percentage`, `requestCount`
- Dark card background with rounded corners
- Provider icon (colored circle with first letter)

**`AnimatedStream.tsx`** — SVG path with flowing animation:
- Props: `path` (SVG d string), `color`, `thickness`, `animated`
- CSS keyframe animation for the flow effect

**`StatsBar.tsx`** — Horizontal metrics row:
- Props: `requests`, `successRate`, `avgLatency`, `fallbacks`
- Light gray labels, bold white values

#### 5.5 Simulation Controls (integrated into LiveTester)

Below the flow diagram or in a collapsible section:
- "Run Simulation" button with config:
  - Number of requests (input, default 100)
  - Mode: Mock / Real (radio)
  - Failure injection sliders (per deployment, 0-100%)
- On simulation complete: update all stats, stream thicknesses, percentages, and request counts
- Loading state: show spinner/progress on the flow diagram

#### 5.6 Data Flow

```typescript
// API response shape the component expects:
interface LiveTesterData {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  avg_latency_ms: number;
  fallback_count: number;
  deployments: {
    deployment_id: string;
    provider: string;         // "nebius", "fireworks_ai", "azure"
    display_name: string;     // "Nebius", "Fireworks", "Azure"
    priority?: number;        // For ordered mode
    weight?: number;          // For weighted mode
    request_count: number;
    success_count: number;
    failure_count: number;
    avg_latency_ms: number;
    percent_of_total: number; // 0-100
  }[];
  // For ordered mode: the cascade flow
  flow_steps?: {
    from_deployment: string | null;
    to_deployment: string;
    request_count: number;
    reason: "primary" | "fallback_error" | "fallback_rate_limit";
  }[];
}
```

### Files to Create

| File | Action |
|------|--------|
| `ui/litellm-dashboard/src/components/routing_groups/LiveTester.tsx` | **NEW** — Main live tester container |
| `ui/litellm-dashboard/src/components/routing_groups/OrderedFallbackFlow.tsx` | **NEW** — Ordered fallback visualization |
| `ui/litellm-dashboard/src/components/routing_groups/WeightedRoundRobinFlow.tsx` | **NEW** — Weighted fan-out visualization |
| `ui/litellm-dashboard/src/components/routing_groups/DeploymentCard.tsx` | **NEW** — Provider deployment card |
| `ui/litellm-dashboard/src/components/routing_groups/AnimatedStream.tsx` | **NEW** — SVG animated flowing stream |
| `ui/litellm-dashboard/src/components/routing_groups/StatsBar.tsx` | **NEW** — Metrics bar (requests, success, latency, fallbacks) |
| `ui/litellm-dashboard/src/components/routing_groups/SimulationControls.tsx` | **NEW** — Simulation config form |

### Dependencies
- Agent 3 (test/simulate API must return the right data shapes)
- Agent 4 (page structure and routing group detail view must exist)

---

## Agent 6: UI — Logs Routing Decision Section

**Goal:** Add a "Routing Decision" section to the Logs detail drawer that shows how the request was routed through a routing group — which deployments were considered, which one handled it, and whether any fallbacks fired.

### Why This Matters

When routing group metadata flows into spend logs (Fix 5), the Logs page already shows the final `model` and `model_id`. But users can't see:
- That the request went through a routing group (vs. a direct model call)
- What strategy was used
- Whether fallbacks were triggered (and which deployments failed before the final one succeeded)
- The full routing trace chain

This section surfaces that information directly in the existing log detail view.

### Design — "Routing Decision" Card

This is a new card section in `LogDetailContent.tsx`, inserted **between "Request Details" and "Metrics"** (only shown when `metadata.routing_group_name` exists):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Routing Decision                                                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │  Routing Group     prod-model                                          │  │
│  │  Strategy          Priority Failover                                   │  │
│  │  Routing Group ID  a1b2c3d4-e5f6-...          [Copy]                   │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Routing Trace ────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │  ① Nebius / meta-llama/Llama-3.3-70B                                   │  │
│  │     Priority 1 · model_id: abc123                                      │  │
│  │     ✗ FAILED  ──  RateLimitError: 429 Too Many Requests  ──  23ms      │  │
│  │                                                                        │  │
│  │     ──── fallback ────▶                                                │  │
│  │                                                                        │  │
│  │  ② Fireworks AI / llama-v3p3-70b                                       │  │
│  │     Priority 2 · model_id: def456                                      │  │
│  │     ✓ SUCCESS  ──  200 OK  ──  225ms                                   │  │
│  │                                                                        │  │
│  │  ────────────────────────────────────────────────────────────────────   │  │
│  │  Azure / gpt-4o (Priority 3) was not tried                             │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ No Fallbacks (direct success) ────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │  ① Nebius / meta-llama/Llama-3.3-70B                                   │  │
│  │     Priority 1 · model_id: abc123                                      │  │
│  │     ✓ SUCCESS  ──  200 OK  ──  149ms                                   │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ Weighted (no fallback chain) ─────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │  Selected: Nebius / meta-llama/Llama-3.3-70B                           │  │
│  │  Strategy: Weighted (83% weight) · model_id: abc123                    │  │
│  │  ✓ SUCCESS  ──  200 OK  ──  149ms                                      │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Three Display Modes

**Mode A: Fallback triggered (priority-failover, at least one failure)**
Shows the numbered step-by-step trace:
- Each attempted deployment in order, with circled number
- Failed steps: red `✗ FAILED` + error class + error message + latency
- `──── fallback ────▶` arrow between failed and next attempt
- Successful step: green `✓ SUCCESS` + status + latency
- Untried deployments: grayed out "(was not tried)" at bottom

**Mode B: Direct success (priority-failover, first deployment succeeded)**
Simplified single-step view:
- Shows the deployment that handled it
- Green `✓ SUCCESS` + latency
- No fallback arrows needed

**Mode C: Non-ordered strategy (weighted, round-robin, cost-based, etc.)**
Shows which deployment was selected:
- "Selected:" label + deployment name
- Strategy name + weight/reason
- Success/failure status + latency

### Data Source

All data comes from fields already in the log entry's `metadata` object (populated by Fix 5):

```typescript
// From metadata (StandardLoggingMetadata):
metadata.routing_group_id      // "a1b2c3d4-..."
metadata.routing_group_name    // "prod-model"
metadata.routing_strategy      // "priority-failover"

// Already in the log entry:
logEntry.model                 // Final model that handled it
logEntry.model_id              // Final deployment ID
logEntry.custom_llm_provider   // Final provider

// From metadata (retries/fallbacks):
metadata.attempted_retries     // How many retries before success
metadata.model_map_information // Which model was mapped
```

**For the full routing trace (failed attempts before success):** The `routing_trace` data needs to be added to `StandardLoggingMetadata` (Agent 1). Add:

```python
class StandardLoggingMetadata(StandardLoggingUserAPIKeyMetadata):
    # ... existing fields ...
    routing_group_id: Optional[str]
    routing_group_name: Optional[str]
    routing_strategy: Optional[str]
    routing_trace: Optional[List[dict]]    # NEW — [{deployment_id, model, provider, status, error, latency_ms}]
```

Each entry in `routing_trace` is a dict:
```python
{
    "deployment_id": "abc123",
    "model": "nebius/meta-llama/Llama-3.3-70B",
    "provider": "nebius",
    "status": "error",           # "success" or "error"
    "error_class": "RateLimitError",
    "error_message": "429 Too Many Requests",
    "latency_ms": 23,
    "priority": 1,               # for priority-failover
    "weight": null,              # for weighted
}
```

Agent 2 wires this: in the Router's fallback handler (`litellm/router_utils/fallback_event_handlers.py`), append each failed attempt to `metadata["routing_trace"]` before trying the next fallback. On final success, the complete trace is in the metadata and flows to spend logs.

### Tasks

#### 6.1 Extend StandardLoggingMetadata (Agent 1 addition)

Add `routing_trace: Optional[List[dict]]` to `StandardLoggingMetadata` in `litellm/types/utils.py`.

#### 6.2 Wire routing_trace in fallback handler (Agent 2 addition)

In `litellm/router_utils/fallback_event_handlers.py`, when a deployment fails and fallback is triggered:
```python
# Append failed attempt to routing_trace
_metadata = kwargs.get("metadata", {})
if "routing_trace" not in _metadata:
    _metadata["routing_trace"] = []
_metadata["routing_trace"].append({
    "deployment_id": kwargs.get("model_id", ""),
    "model": kwargs.get("model", ""),
    "provider": kwargs.get("custom_llm_provider", ""),
    "status": "error",
    "error_class": type(exception).__name__,
    "error_message": str(exception)[:200],
    "latency_ms": round((end_time - start_time).total_seconds() * 1000),
    "priority": metadata.get("deployment_priority"),
    "weight": metadata.get("deployment_weight"),
})
```

On final success, the success callback appends the final entry with `"status": "success"`.

#### 6.3 RoutingDecisionSection Component

Create `ui/litellm-dashboard/src/components/view_logs/LogDetailsDrawer/RoutingDecisionSection.tsx`:

```typescript
interface RoutingDecisionSectionProps {
  metadata: Record<string, any>;
  logEntry: LogEntry;
}
```

**Rendering logic:**
1. Check `metadata.routing_group_name` — if missing, don't render (request didn't go through a routing group)
2. Render header card: Routing Group name, Strategy badge, Group ID with copy
3. Check `metadata.routing_trace`:
   - If array with entries: render Mode A (fallback trace) or Mode B (single success)
   - If missing/empty and strategy is not priority-failover: render Mode C (selected deployment)
4. Style: use Ant Design `Card`, `Tag`, `Descriptions`, `Steps` (vertical) or custom layout
   - Failed step: red border-left, `Tag color="error"` for status
   - Success step: green border-left, `Tag color="success"` for status
   - Fallback arrow: gray dashed divider with "fallback" label
   - Untried deployments: grayed out text

#### 6.4 Integrate into LogDetailContent

Edit `ui/litellm-dashboard/src/components/view_logs/LogDetailsDrawer/LogDetailContent.tsx`:

Insert between "Request Details" card and "Metrics" section:
```tsx
{/* Routing Decision (only for routing group requests) */}
{metadata?.routing_group_name && (
  <RoutingDecisionSection metadata={metadata} logEntry={logEntry} />
)}

{/* Metrics */}
<MetricsSection logEntry={logEntry} metadata={metadata} />
```

### Files to Create/Modify

| File | Action |
|------|--------|
| `litellm/types/utils.py` | Add `routing_trace` to `StandardLoggingMetadata` (Agent 1 scope) |
| `litellm/router_utils/fallback_event_handlers.py` | Wire `routing_trace` on failures (Agent 2 scope) |
| `ui/litellm-dashboard/src/components/view_logs/LogDetailsDrawer/RoutingDecisionSection.tsx` | **NEW** — Routing decision card |
| `ui/litellm-dashboard/src/components/view_logs/LogDetailsDrawer/LogDetailContent.tsx` | Insert RoutingDecisionSection |

### Dependencies
- Agent 1 (types: `routing_trace` field in StandardLoggingMetadata)
- Agent 2 (wiring: populate `routing_trace` in fallback handler)
- Can be built in parallel with Agents 4+5 (separate UI area)

---

## Agent Dependency Graph

```
Agent 1 (DB & Types)
   │
   ├──→ Agent 2 (API Endpoints)
   │       │
   │       ├──→ Agent 3 (Test Engine)
   │       │       │
   │       │       └──→ Agent 5 (Live Tester UI)
   │       │
   │       ├──→ Agent 4 (Builder UI + Health Tab)
   │       │       │
   │       │       └──→ Agent 5 (Live Tester UI)
   │       │
   │       └──→ Agent 6 (Logs Routing Decision UI)
   │
   └──→ Agent 4 (Builder UI) [can start types/networking in parallel]
```

**Parallelism opportunities:**
- Agent 1 runs first (small, fast)
- Agents 2 + 4 can start in parallel once Agent 1 is done (API contracts agreed upfront)
- Agent 3 can start once Agent 2's endpoint signatures are defined
- Agent 5 starts once Agents 3 + 4 are done
- Agent 6 can run in parallel with Agents 4 + 5 (separate UI area — Logs page vs. Routing Groups page)

---

## Testing Strategy

### Unit Tests (Agent per agent)

| Agent | Test File | What to Test |
|-------|-----------|--------------|
| 1 | `tests/litellm/proxy/test_routing_group_types.py` | Pydantic model validation, serialization |
| 2 | `tests/litellm/proxy/test_routing_group_endpoints.py` | CRUD operations, router sync logic |
| 3 | `tests/litellm/proxy/test_routing_group_test_engine.py` | Single request test, simulation, failure injection, trace callback |
| 4 | `ui/litellm-dashboard/src/components/routing_groups/__tests__/` | Component rendering, form validation |
| 5 | `ui/litellm-dashboard/src/components/routing_groups/__tests__/` | Chart rendering, simulation result display |
| 6 | `ui/litellm-dashboard/src/components/view_logs/__tests__/` | RoutingDecisionSection rendering with/without routing data |

### Integration Tests

| Test | Description |
|------|-------------|
| `tests/proxy_unit_tests/test_routing_group_e2e.py` | Create group → test request → verify routing path |
| `tests/proxy_unit_tests/test_routing_group_failover.py` | Priority failover with injected failures |
| `tests/proxy_unit_tests/test_routing_group_access_control.py` | Team/key access restriction |

### E2E Tests

| Test | Description |
|------|-------------|
| `ui/litellm-dashboard/e2e_tests/tests/routingGroups/createRoutingGroup.spec.ts` | Full builder flow |
| `ui/litellm-dashboard/e2e_tests/tests/routingGroups/testRoutingGroup.spec.ts` | Live tester flow |

---

## Summary: Files by Agent

### Agent 1: Database & Types
```
litellm/proxy/schema.prisma                          (modify)
litellm/types/router.py                              (modify)
litellm/types/utils.py                               (modify — add routing_group fields to StandardLoggingMetadata)
tests/litellm/proxy/test_routing_group_types.py      (new)
```

### Agent 2: API Endpoints
```
litellm/proxy/management_endpoints/routing_group_endpoints.py  (new)
litellm/proxy/proxy_server.py                                  (modify — register endpoints, config drift protection, startup sync)
litellm/proxy/management_endpoints/model_management_endpoints.py (modify — stale ref check on delete)
litellm/router.py                                              (modify — model_group_routing_strategy, routing_group_configs)
tests/litellm/proxy/test_routing_group_endpoints.py            (new)
tests/proxy_unit_tests/test_routing_group_e2e.py               (new)
```

### Agent 3: Test Engine
```
litellm/proxy/routing_group_utils/__init__.py                  (new)
litellm/proxy/routing_group_utils/test_engine.py               (new)
litellm/proxy/routing_group_utils/routing_trace_callback.py    (new)
tests/litellm/proxy/test_routing_group_test_engine.py          (new)
```

### Agent 4: UI Builder + Health Tab
```
ui/litellm-dashboard/src/app/(dashboard)/routing-groups/page.tsx              (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsView.tsx      (new — TabGroup with 3 tabs)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsTable.tsx     (new — Tab 1)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupBuilder.tsx    (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingFlowDiagram.tsx     (new)
ui/litellm-dashboard/src/components/routing_groups/DeploymentSelector.tsx     (new)
ui/litellm-dashboard/src/components/routing_groups/StrategySelector.tsx       (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupHealthDashboard.tsx (new — Tab 2)
ui/litellm-dashboard/src/components/networking.tsx                            (modify)
ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts           (new)
ui/litellm-dashboard/src/app/(dashboard)/components/Sidebar2.tsx             (modify)
```

### Agent 5: UI Live Tester (Tab 3)
```
ui/litellm-dashboard/src/components/routing_groups/LiveTester.tsx             (new)
ui/litellm-dashboard/src/components/routing_groups/TrafficSimulator.tsx       (new)
ui/litellm-dashboard/src/components/routing_groups/TrafficDistributionChart.tsx (new)
ui/litellm-dashboard/src/components/routing_groups/SimulationFlowDiagram.tsx  (new)
ui/litellm-dashboard/src/components/routing_groups/RequestFlowTrace.tsx       (new)
```

### Agent 6: UI Logs Routing Decision
```
ui/litellm-dashboard/src/components/view_logs/LogDetailsDrawer/RoutingDecisionSection.tsx  (new)
ui/litellm-dashboard/src/components/view_logs/LogDetailsDrawer/LogDetailContent.tsx        (modify — insert RoutingDecisionSection)
litellm/types/utils.py                                                                     (modify — add routing_trace, Agent 1 scope)
litellm/router_utils/fallback_event_handlers.py                                            (modify — wire routing_trace, Agent 2 scope)
```
