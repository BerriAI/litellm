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
