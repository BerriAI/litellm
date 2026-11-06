# LiteFlow gateway workflows

The gateway accepts LiteFlow compiled contract version `1` and executes sequential `chat_completion` and `python` nodes. Python nodes use the Monty adapter. Conditional, parallel and streaming execution are not supported

Workflow definitions belong to a team. Authors register and edit drafts. A team administrator who did not contribute to the draft publishes an immutable snapshot. Invocation requires a virtual key assigned to that team, with access to the workflow model and each child model

## SDK example

The proxy imports the actual LiteFlow SDK models and digest helper, pinned to SDK revision `27bcbce4`. Its compiled JSON is validated at the server boundary because the SDK does not export a server bundle parser or executor. No SDK implementation is copied

This contract matches LiteFlow PRs [3](https://github.com/BerriAI/liteflow/pull/3) and [5](https://github.com/BerriAI/liteflow/pull/5). Configure `gateway-model` on the proxy and provide two different team members' keys

```python
import os
import time

from liteflow import ChatCompletion, GatewayClient, Input, Workflow

workflow = Workflow(
    name="welcome",
    nodes=[ChatCompletion(model="gateway-model", messages=Input.messages)],
)
messages = [{"role": "user", "content": "Write one short welcome sentence."}]

with GatewayClient(
    base_url=os.environ["PROXY_BASE_URL"],
    api_key=os.environ["WORKFLOW_AUTHOR_KEY"],
    team_id=os.environ["WORKFLOW_TEAM_ID"],
) as author, GatewayClient(
    base_url=os.environ["PROXY_BASE_URL"],
    api_key=os.environ["WORKFLOW_PUBLISHER_KEY"],
    team_id=os.environ["WORKFLOW_TEAM_ID"],
) as publisher:
    draft = author.register(workflow)
    author.validate(draft)
    publisher.publish(draft)
    accepted = author.test(draft, messages=messages, authorize_billing=True)
    for _ in range(300):
        trace = author.inspect(accepted["run_id"])
        print(trace)
        if trace.state.value in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    result = author.run("welcome", messages=messages)
    print(result.run_id, result.output)
```

## API contract

| Operation | Endpoint | Contract |
| --- | --- | --- |
| Register | `POST /v1/workflows` | `{"bundle": <compiled bundle>}`, optional `team_id` query, returns `201` |
| List definitions | `GET /v1/workflows` | `team_id`, `limit`, `cursor`; returns `workflows`, `next_cursor` |
| Retrieve draft | `GET /v1/workflows/{workflow_id}` | Returns the draft and `ETag` header |
| Replace draft | `PUT /v1/workflows/{workflow_id}` | Compiled bundle and exact `If-Match` |
| Validate | `POST /v1/workflows/{workflow_id}/validate` | Exact `If-Match`, no body required |
| Publish | `POST /v1/workflows/{workflow_id}/publish` | Exact `If-Match` and `{"digest": "<draft digest>"}` |
| Test draft | `POST /v1/workflows/{workflow_id}/test` | Exact `If-Match`, `messages`, `authorize_billing: true`; returns `202` |
| Invoke published | `POST /v1/chat/completions` | `model: "workflow/<name>"`, `messages`, `stream: false` |
| List runs | `GET /v1/workflows/{workflow_id}/runs` | `limit`, `status`, `after`; returns `runs`, `count`, `next_cursor` |
| Inspect | `GET /v1/workflows/runs/{run_id}/steps` | Run state and typed step traces |
| Pinned definition | `GET /v1/workflows/runs/{run_id}/definition` | Exact snapshot used by that run |
| Cancel | `POST /v1/workflows/runs/{run_id}/cancel` | Returns `202`; poll until terminal state |

Draft responses contain `name`, `workflow_id`, `team_id`, `etag`, `digest`, `bundle` and `published_snapshot_id`. Missing or stale `If-Match` returns `412`. Validation checks the compiled graph and Python program without executing model or Python nodes

Test runs and chat invocations accept `Idempotency-Key`. Retrying the same operation with the same key and input returns the original run. Reusing that key with different input returns `409`. A test run pins the exact draft revision even if the draft changes later

For live UI progress, send `Prefer: respond-async` on a published chat invocation. The gateway returns `202` with `run_id`, `snapshot_id`, `trace_url` and the `x-litellm-workflow-run-id` header. The ordinary SDK call waits for a Chat Completions response carrying the same run-ID header. It waits until execution finishes or the workflow deadline expires; clients with shorter HTTP timeouts should use asynchronous invocation and an idempotency key

## Trace and spend data

Draw the pinned bundle's `nodes` and resolve each binding's `from` reference. Match `step.node_id` to `node.id`. Run and node states are `pending`, `running`, `completed`, `failed` and `cancelled`

Each step exposes typed `input` and `output`, `attempts`, `started_at`, `finished_at`, `deadline`, sanitized `error`, token counts and `response_cost`. A run-level `error` explains failures before any node starts

Model steps carry a deterministic `call_id` before dispatch. `request_id` and `request_ids` link to actual `LiteLLM_SpendLogs.request_id` values. Spend writes are asynchronous, so receipt IDs can be absent on the first read and appear on later polls. The trace reader sums persisted attempts once and does not duplicate earlier checkpoints

Each child uses the standard gateway authorization, hooks, guardrails, limits and billing path. The workflow wrapper emits no additional bill. Python nodes have no model spend receipt. Known costs remain visible when output validation fails

Definitions and snapshots use separate tables. Execution reuses `LiteLLM_WorkflowRun` and `LiteLLM_WorkflowEvent`. Engine-owned runs are exposed only through the scoped execution endpoints; existing tracking endpoints continue to serve externally tracked runs

## Execution limits and interruption

The initial runner allows four active runs per worker and 64 pending or running runs per team. The request that creates the durable run owns its execution task; idempotent retries observe that same run. Workers drain for up to 30 seconds on graceful shutdown, then fail unfinished work. After a crash, unfinished runs fail at their original deadline when read or during subsequent admission. Stored traces survive; executions are never replayed automatically

Chat retries are limited to rate limits confirmed before provider dispatch. Provider failures, timeouts and attempts with receipts are terminal because replaying an uncertain request could charge twice

Cancellation is cooperative. It stops downstream nodes but cannot undo a model request already sent. Trace receipts can still arrive after cancellation

Ordinary virtual keys can read their own runs. Team administrators and proxy administrators can inspect team runs within their credential scope. Definition access does not grant another caller access to private run inputs or outputs
