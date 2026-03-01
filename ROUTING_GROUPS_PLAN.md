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

#### 1.3 Migration

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

**For "priority-failover" strategy:**
```python
# Deployments ordered by priority
# Primary = priority 1, fallbacks = priority 2, 3, ...
# Map to: model_group = routing_group_name, all deployments share this model_name
# Fallbacks configured as: [{routing_group_name: [fallback_group_2, fallback_group_3]}]
#
# Actually simpler: use a SINGLE model group with all deployments,
# but set the routing_strategy to use priority-based selection.
# The Scheduler class already supports priority-based scheduling.
```

**Implementation approach:**
- For **priority-failover**: Create separate model groups per priority tier, wire them as `fallbacks`. E.g., `routing_group_name` has only priority-1 deployments. `routing_group_name_fallback_2` has priority-2 deployments. Fallback chain: `[{routing_group_name: [routing_group_name_fallback_2, routing_group_name_fallback_3]}]`
- For **simple-shuffle / least-busy / latency-based / cost-based / usage-based**: Create a single model group with all deployments, set `routing_strategy` on router or use per-group strategy override
- For **weighted**: Create a single model group, set `weight` on each deployment's `litellm_params`

#### 2.3 Sync on Startup

In `ProxyConfig.load_config()` and `ProxyConfig.add_deployment()`:
- After loading DB models, also load routing groups from `LiteLLM_RoutingGroupTable`
- Call `_sync_routing_group_to_router()` for each active routing group

#### 2.4 Access Control Integration

When a routing group has `assigned_team_ids` / `assigned_key_ids`:
- Automatically add the `routing_group_name` to the team's `models` list (or use access groups)
- On routing group deletion, clean up these references

### Files to Create/Modify

| File | Action |
|------|--------|
| `litellm/proxy/management_endpoints/routing_group_endpoints.py` | **NEW** — CRUD + test + simulate |
| `litellm/proxy/proxy_server.py` | Register new router, add sync-on-startup |
| `litellm/proxy/management_endpoints/internal_user_endpoints.py` | Optional: add routing_group_ids to user model |
| `litellm/router.py` | Possibly add helper for per-group strategy override |

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

#### 4.2 Routing Groups List View

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

### Files to Create/Modify

| File | Action |
|------|--------|
| `ui/litellm-dashboard/src/app/(dashboard)/routing-groups/page.tsx` | **NEW** — Page component |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsView.tsx` | **NEW** — Main view |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsTable.tsx` | **NEW** — List table |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingGroupBuilder.tsx` | **NEW** — Create/Edit form |
| `ui/litellm-dashboard/src/components/routing_groups/RoutingFlowDiagram.tsx` | **NEW** — Visual flow |
| `ui/litellm-dashboard/src/components/routing_groups/DeploymentSelector.tsx` | **NEW** — Model picker |
| `ui/litellm-dashboard/src/components/routing_groups/StrategySelector.tsx` | **NEW** — Strategy picker |
| `ui/litellm-dashboard/src/components/networking.tsx` | Add CRUD + test/simulate calls |
| `ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts` | **NEW** — React Query hooks |
| `ui/litellm-dashboard/src/app/(dashboard)/components/Sidebar2.tsx` | Add "Routing Groups" menu item |

### Dependencies
- Agent 2 (API endpoints must be defined)

---

## Agent 5: UI — Live Tester & Traffic Simulator

**Goal:** Build the testing and simulation UI that lets users see live request flow and traffic distribution.

### Tasks

#### 5.1 Live Tester Tab (within Routing Group detail view)

A tab on the routing group detail/edit page:

```
┌─────────────────────────────────────────────────────────────┐
│  Routing Group: "My LLM Pipeline"                            │
│                                                              │
│  [Overview] [Live Tester] [Traffic Simulator] [Settings]     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ── Send a Test Request ───────────────────────────────────  │
│                                                              │
│  Message: [Hello, respond with one word.        ]            │
│                                                              │
│  [  Send Test Request  ]    [  Send (Mock)  ]                │
│                                                              │
│  ── Request Flow ──────────────────────────────────────────  │
│                                                              │
│  ✓ Request sent to: Nebius / meta-llama/Llama-3.3-70B        │
│    → Status: Success (234ms)                                 │
│    → Response: "Hello"                                       │
│                                                              │
│  OR (with fallback):                                         │
│                                                              │
│  ✗ Request sent to: Nebius / meta-llama/Llama-3.3-70B        │
│    → Status: Error (1203ms) — "RateLimitError: 429"          │
│  ↓ Fallback triggered                                        │
│  ✓ Request sent to: Fireworks AI / llama-v3p3-70b            │
│    → Status: Success (189ms)                                 │
│    → Response: "Hello"                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 5.2 Traffic Simulator Tab

```
┌─────────────────────────────────────────────────────────────┐
│  ── Simulation Config ─────────────────────────────────────  │
│                                                              │
│  Number of requests: [100]                                   │
│  Concurrency:        [10 ]                                   │
│  Mode: ○ Mock (fast, no real LLM calls)                      │
│        ○ Real (actual LLM calls, costs money)                │
│                                                              │
│  ── Failure Injection ─────────────────────────────────────  │
│                                                              │
│  Nebius:       [  50]% failure rate   ← slider               │
│  Fireworks AI: [   0]% failure rate                          │
│  Azure:        [   0]% failure rate                          │
│                                                              │
│  [  Run Simulation  ]                                        │
│                                                              │
│  ── Traffic Distribution ──────────────────────────────────  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Traffic Split (Bar Chart)               │     │
│  │                                                      │     │
│  │  Nebius        ████████████████░░░░ 50 (25 ok, 25 err)│   │
│  │  Fireworks AI  ██████████████████░░ 40 (38 ok, 2 err) │   │
│  │  Azure         ████░░░░░░░░░░░░░░░ 10 (10 ok, 0 err) │   │
│  │                                                      │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ── Flow Diagram (for Priority Failover) ──────────────────  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                                                      │     │
│  │  [100 requests]                                      │     │
│  │       │                                              │     │
│  │       ▼                                              │     │
│  │  ┌─────────┐   50 ok    ┌──────────┐               │     │
│  │  │ Nebius  │──────────→│ Response  │               │     │
│  │  └─────────┘            └──────────┘               │     │
│  │       │ 50 errors                                    │     │
│  │       ▼                                              │     │
│  │  ┌───────────┐  38 ok   ┌──────────┐               │     │
│  │  │ Fireworks │─────────→│ Response  │               │     │
│  │  └───────────┘           └──────────┘               │     │
│  │       │ 12 errors                                    │     │
│  │       ▼                                              │     │
│  │  ┌─────────┐   10 ok    ┌──────────┐               │     │
│  │  │ Azure   │──────────→│ Response  │               │     │
│  │  └─────────┘            └──────────┘               │     │
│  │       │ 2 errors                                     │     │
│  │       ▼                                              │     │
│  │  ┌─────────┐                                        │     │
│  │  │ Failed  │ 2 requests                             │     │
│  │  └─────────┘                                        │     │
│  │                                                      │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ── Summary ───────────────────────────────────────────────  │
│                                                              │
│  Total: 100 | Succeeded: 98 (98%) | Failed: 2 (2%)          │
│  Avg Latency: 312ms | P50: 245ms | P99: 1,203ms             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 5.3 Traffic Distribution Chart Component

Use Tremor's `BarChart` (already available in the project) to show:

- **Horizontal stacked bar chart**: Each deployment gets a bar. Green = success, Red = failure.
- **Labels**: Deployment name, provider logo, count, percentage.

For **priority-failover**: Also render the **flow diagram** (vertical Sankey-like) showing:
- Entry → Primary → (success → exit) or (error → Fallback 1) → (success → exit) or (error → Fallback 2) → ...

Implementation: CSS flex/grid layout with styled divs and arrows. No need for a heavy charting library.

#### 5.4 Real-Time Progress (for simulation)

During simulation (which may take several seconds for 100+ mock requests):
- Show a progress bar
- Update the traffic distribution chart incrementally as results come in
- Option: Use SSE (Server-Sent Events) from the `/simulate` endpoint to stream results, OR
- Simpler option: Poll with the simulation running server-side, return all results at once (simpler to implement)

**Recommendation:** Start with the simpler approach (return all results at once). The mock simulation of 100 requests should complete in < 1 second anyway.

### Files to Create

| File | Action |
|------|--------|
| `ui/litellm-dashboard/src/components/routing_groups/LiveTester.tsx` | **NEW** — Single request tester |
| `ui/litellm-dashboard/src/components/routing_groups/TrafficSimulator.tsx` | **NEW** — Bulk simulation UI |
| `ui/litellm-dashboard/src/components/routing_groups/TrafficDistributionChart.tsx` | **NEW** — Bar chart |
| `ui/litellm-dashboard/src/components/routing_groups/SimulationFlowDiagram.tsx` | **NEW** — Sankey-like flow |
| `ui/litellm-dashboard/src/components/routing_groups/RequestFlowTrace.tsx` | **NEW** — Single-request trace view |

### Dependencies
- Agent 3 (test/simulate API must return the right data shapes)
- Agent 4 (page structure and routing group detail view must exist)

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
   │       └──→ Agent 4 (Builder UI)
   │               │
   │               └──→ Agent 5 (Live Tester UI)
   │
   └──→ Agent 4 (Builder UI) [can start types/networking in parallel]
```

**Parallelism opportunities:**
- Agent 1 runs first (small, fast)
- Agents 2 + 4 can start in parallel once Agent 1 is done (API contracts agreed upfront)
- Agent 3 can start once Agent 2's endpoint signatures are defined
- Agent 5 starts once Agents 3 + 4 are done

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
tests/litellm/proxy/test_routing_group_types.py      (new)
```

### Agent 2: API Endpoints
```
litellm/proxy/management_endpoints/routing_group_endpoints.py  (new)
litellm/proxy/proxy_server.py                                  (modify)
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

### Agent 4: UI Builder
```
ui/litellm-dashboard/src/app/(dashboard)/routing-groups/page.tsx              (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsView.tsx      (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupsTable.tsx     (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingGroupBuilder.tsx    (new)
ui/litellm-dashboard/src/components/routing_groups/RoutingFlowDiagram.tsx     (new)
ui/litellm-dashboard/src/components/routing_groups/DeploymentSelector.tsx     (new)
ui/litellm-dashboard/src/components/routing_groups/StrategySelector.tsx       (new)
ui/litellm-dashboard/src/components/networking.tsx                            (modify)
ui/litellm-dashboard/src/app/(dashboard)/hooks/useRoutingGroups.ts           (new)
ui/litellm-dashboard/src/app/(dashboard)/components/Sidebar2.tsx             (modify)
```

### Agent 5: UI Live Tester
```
ui/litellm-dashboard/src/components/routing_groups/LiveTester.tsx             (new)
ui/litellm-dashboard/src/components/routing_groups/TrafficSimulator.tsx       (new)
ui/litellm-dashboard/src/components/routing_groups/TrafficDistributionChart.tsx (new)
ui/litellm-dashboard/src/components/routing_groups/SimulationFlowDiagram.tsx  (new)
ui/litellm-dashboard/src/components/routing_groups/RequestFlowTrace.tsx       (new)
```
