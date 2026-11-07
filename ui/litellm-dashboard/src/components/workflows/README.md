# Workflow definition components

`WorkflowDefinitionView` combines a read-only graph and definition inspector. `WorkflowGraph` and `WorkflowDefinitionInspector` are also exported separately from `@/components/workflows`

Pass the `bundle` field from the LiteFlow PR #3 `Draft` response directly. The component validates the compiler's version 1 shape at its boundary, draws nodes in array order and derives connections from adjacent nodes. It does not fetch, register, edit or execute workflows

```tsx
import { WorkflowDefinitionView } from "@/components/workflows";

const definitionKey = `${draft.workflow_id}:${draft.etag}`;

<WorkflowDefinitionView
  key={definitionKey}
  bundle={draft.bundle}
  definitionKey={definitionKey}
  selectedNodeId={selectedNodeId}
  onNodeSelect={setSelectedNodeId}
  loading={isLoading}
  error={loadError}
/>;
```

Keep fetching, error recovery and selection state in the parent. Reset selection when changing definitions, or key the parent that owns selection by the definition identity. `onNodeSelect` receives the compiled node ID, or `null` when the user clears selection. Run details and Test run controls can use the same selection without owning a second graph

For a run, fetch its pinned compiled definition and use its immutable `snapshot_id` as `definitionKey`. Map the typed PR #5 step states by `step.node_id`, which must equal the compiled `node.id`

```tsx
const execution = {
  definitionKey: steps.snapshot_id,
  states: Object.fromEntries(steps.steps.map((step) => [step.node_id, step.state])),
};

<WorkflowDefinitionView
  bundle={pinnedBundle}
  definitionKey={snapshotId}
  selectedNodeId={selectedNodeId}
  onNodeSelect={setSelectedNodeId}
  execution={execution}
/>;
```

An overlay from a different definition is ignored. Missing states leave nodes uncolored. The graph supports `pending`, `running`, `completed`, `failed` and `cancelled`. Resolve any repeated step records to the current node state in the parent before constructing the map. Workflow drafts should use their workflow ID plus ETag and must not borrow a run's overlay without its pinned bundle

The inspector displays compiler bindings, Python schemas and source, structured ChatCompletion output and execution settings. It never executes Python or renders definition content as HTML. Parallel and If nodes and unknown bundle versions produce an unsupported state instead of misleading sequential edges

The UI uses the dashboard's shadcn Button, Card, Badge, Collapsible and Skeleton components, shared theme tokens and the existing theme provider. It follows light and dark mode without a second theme toggle. Drag or scroll the canvas to pan, use Ctrl or Command plus scroll to zoom, or use the toolbar. Keyboard users can tab to nodes and controls, pan the focused canvas with arrow keys, zoom with plus/minus and fit with zero

## Fixture preview and verification

Fixtures and their generator live in `tests/fixtures/workflows`. The development route `/workflows/graph-preview` returns not found outside development. Fixtures are never a production fallback

From `ui/litellm-dashboard`, start the local shell fixture service and the actual dashboard in separate terminals

```sh
../../scripts/with_dashboard_node.sh node tests/fixtures/workflows/serve-shell.mjs
```

```sh
NEXT_PUBLIC_USE_REWRITES=true LENS_DEV_PROXY_URL=http://127.0.0.1:3368 \
  ../../scripts/with_dashboard_node.sh npm run dev:webpack -- --hostname 127.0.0.1 --port 3367
```

Open `http://workflow-graph.localhost:3367/__workflow_fixture_login`. The isolated hostname keeps the synthetic session separate from other localhost dashboard sessions. The fixture service binds only to loopback, supplies dashboard shell responses and rejects unconfigured requests. It implements no workflow execution or persistence

Select the Python node to inspect its literal and previous-step bindings, schemas, source, timeout and retries. Select the last ChatCompletion to inspect structured output. Use the fixture preview controls for execution overlays and loading, empty and unsupported states. Use the dashboard header to switch themes and verify pan, zoom, fit and keyboard selection

This is fixture-based proof in the real dashboard. It does not establish live gateway registration, execution, run polling, billing or Test run integration

```sh
../../scripts/with_dashboard_node.sh npx vitest run \
  src/components/workflows/definition.test.ts \
  src/components/workflows/viewport.test.ts \
  src/components/workflows/WorkflowDefinitionView.integration.test.tsx
../../scripts/with_dashboard_node.sh npm run typecheck
```
