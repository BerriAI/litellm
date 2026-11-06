# Workflow run debugging

Reusable history and trace components for the workflow page. The development preview uses synthetic API responses and does not execute or bill workflows

## Public interfaces

Import from `@/components/workflows/runs`

| Export                  | Purpose                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `createWorkflowRunApi`  | Authenticated HTTP adapter with runtime response validation  |
| `WorkflowRunHistory`    | Workflow-scoped history, status filter and cursor pagination |
| `WorkflowRunInspector`  | Controlled presentation of a trace and its pinned definition |
| `WorkflowRunDebugger`   | History, selected-run queries, polling and graph callbacks   |
| `WorkflowRunDataSource` | Replaceable run-list, trace and pinned-definition boundary   |
| `WorkflowRunGraphState` | Pinned definition, all step executions and selected step     |
| `WorkflowStepSelection` | `runId`, `snapshotId`, `stepId` and `nodeId`                 |

The barrel also exports request and response types from `contracts.ts`. Keep one data-source instance per backend, team and authenticated session. Recreate it when any of those change. Its opaque `cacheKey` scopes query caches and must never contain a bearer token

```tsx
const dataSource = useMemo(() => createWorkflowRunApi({ accessToken, teamId }), [accessToken, teamId, backendIdentity]);

<WorkflowRunDebugger
  key={workflowId}
  workflowId={workflowId}
  dataSource={dataSource}
  runId={selectedRunId}
  onSelectRun={setSelectedRunId}
  onGraphStateChange={setGraphState}
/>;
```

The owning page controls the selected run and any URL state through nuqs. Clear its selected run when switching workflows. Local filters and pagination reset when the workflow, data source or page size changes. `selectedStepId` and `onSelectStep` support selection controlled by the graph. Without a controlled step, the inspector initially selects the failed step, then the first reported step

## Confirmed trace contract

`RunSteps` and `StepTrace` match `src/liteflow/client.py` at LiteFlow PR #5 commit `27bcbce4e84fb73b1d4bdf2fa108df53e9b50c2d`

`GET /v1/workflows/runs/{run_id}/steps` returns `run_id`, `snapshot_id`, `state`, `cancel_requested` and `steps`

Each step preserves `step_id`, `run_id`, `node_id`, `state`, JSON `input` and `output`, `attempts`, `deadline`, `started_at`, nullable `finished_at`, nullable `error`, `prompt_tokens`, `completion_tokens`, nullable `response_cost` and nullable `request_id`

Step IDs must be unique within a run. Multiple executions may reference the same node. The graph callback retains every execution rather than using node IDs as step keys. Null, false, zero and empty strings remain distinct JSON values. Unknown costs show “Not reported”. Aggregate cost is labeled “Reported step cost” and marked partial when any step has an unknown cost

## Provisional gateway assumptions

Only `api.ts` understands the provisional wire responses

```text
GET /v1/workflows/runs
  team_id=<team>
  workflow_id=<workflow>
  after=<opaque cursor, omitted on first page>
  status=<state, omitted for all>
  limit=<page size>

{
  "runs": [{
    "run_id": "...",
    "workflow_id": "...",
    "snapshot_id": "...",
    "status": "completed",
    "created_at": "2026-11-06T08:30:00Z"
  }],
  "next_cursor": null
}
```

```text
GET /v1/workflows/runs/{run_id}/definition

{
  "workflow_id": "...",
  "snapshot_id": "...",
  "bundle": { "contract_version": "1", "nodes": [] }
}
```

The adapter rejects mismatched workflow, run and snapshot identities. The pinned definition is queried only after the trace supplies its snapshot ID. No current-draft endpoint is used as a fallback. A missing definition leaves the trace available and passes a null definition to the graph

The first history page refreshes every five seconds. Selected pending and running traces refresh every two seconds, stopping at terminal states. Requests receive React Query abort signals. Selection and cache keys isolate late responses from previous runs or sessions. Poll failures expose a retry action and label displayed data as the last successful response

These reads can be backed by the gateway's existing workflow run and event storage. The frontend neither reads SpendLogs for history nor maintains a separate persistent store

## Exact spend details

“View spend record” appears only when the step has a `request_id`. Opening it uses the dashboard's existing `useLogDetails` authorization flow with that exact ID and the step's original timestamp. The endpoint is `/spend/logs/ui/{request_id}?start_date=...`

This avoids the Logs page's current date-window search and performs no spend aggregation or billing writes. The payload returned by the authorized detail endpoint is shown as JSON. Permission, retention and missing-log failures leave workflow trace inspection available

## Local fixture verification

From `ui/litellm-dashboard`, run these commands in separate terminals

```sh
../../scripts/with_dashboard_node.sh node tests/fixtures/workflow-runs/serve-shell.mjs
```

```sh
LENS_DEV_PROXY_URL=http://127.0.0.1:3378 NEXT_PUBLIC_USE_REWRITES=true \
  ../../scripts/with_dashboard_node.sh npm run dev:webpack -- \
  --hostname 127.0.0.1 --port 3377
```

Open `http://workflow-runs.localhost:3377/__workflow_run_fixture_login`. The preview is development-only, served inside the real dashboard shell, and labeled “Fixture preview”. Its local fixture credential is not a gateway credential

Select the completed, running and failed runs. Inspect Python values, a failed step with two attempts and an LLM spend record. Filter by status, navigate to the second history page, then inspect two distinct executions referencing `node_1`. The pending fixture has no reported steps. The scalar Python fixture deliberately exercises JSON false and zero as rendering edge cases

The preview serves run and definition responses over HTTP through `createWorkflowRunApi`. Active fixtures remain active until changed by the fixture server. There is no simulated completion timer. The integration test separately verifies a gateway response transitioning a run to completion and stopping polling

```sh
../../scripts/with_dashboard_node.sh npx vitest run \
  src/components/workflows/runs/api.test.ts \
  src/components/workflows/runs/WorkflowRunDebugger.integration.test.tsx
```

Stop both local processes after verification. This preview does not establish live gateway integration, real provider execution or authorization against the backend under development
