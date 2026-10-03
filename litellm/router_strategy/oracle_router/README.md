# ORACLE router

Program-level routing for agentic workloads: an agentic task (a *program*) is bound to one model at its
first request, every later request with the same `program_id` reuses that binding, and the model is
chosen by a pluggable decision maker that learns from a pluggable verifier's delayed verdict.
Paper: https://arxiv.org/abs/2607.22465

## Config

```yaml
model_list:
  - model_name: oracle-router
    litellm_params:
      model: auto_router/oracle_router
      oracle_router_config:
        available_models: ["smart", "fast"]
        decision_maker: { type: thompson, weights: { quality: 0.7, cost: 0.3 } }
        verifier: { type: guardrail, guardrail: answer-judge }
```

Clients send `metadata.program_id` (or `litellm_session_id`, or the `x-litellm-program-id` header)
with every request of a task. Requests without any id are routed statelessly and never learn. Concurrent
first requests of one program wait for the decision in flight, so a program is bound exactly once.

## Decision makers

| `type` | binds a program by | learns |
|---|---|---|
| `thompson` (default) | LiteLLM's adaptive-router bandit: a Beta posterior per (request type, model), Thompson sampling, cost-weighted pick with `weights: {quality, cost}` | from verified program scores |
| `pre_routing` | the strategy router named by `router` (complexity, quality, adaptive or semantic), once per program | no |
| `fixed` | always `model` | no |
| `custom` | `path: pkg.mod:Factory`, called with the model tuple; must expose `models`, `async select`, `update`, `snapshot` | up to you |

## Verifiers

| `type` | scores a finished program by |
|---|---|
| `guardrail` | applying the LiteLLM guardrail named `guardrail` to the program's final turn, with the transcript as `structured_messages` and the feedback `payload` as `request_data["metadata"]`: pass = 1, blocked = 0 (read from the status the guardrail records, so an `llm_as_a_judge` with `on_failure: log` still scores 0 when the answer fails). A guardrail that could not evaluate the program, for example a judge whose model was unreachable, counts as a failed verification and teaches the decision maker nothing |
| `reported` (default) | `metadata.program_score` on the last request, or `POST /oracle_router/feedback` |
| `custom` | `path: pkg.mod:Factory`, called with no arguments; must expose `async verify(outcome)` |

A verified score `s` adds `s` successes and `1 - s` failures to the cell of the model that served the
program. Updates are applied in verification-arrival order. A verifier that raises, or returns a score
that is not a finite number, counts as a failed verification (`verifications_failed` in the state) and
never reaches the decision maker.

## Finishing a program

Either mark the last request with `metadata.program_done: true` (plus `program_score` for the reported
verifier), or call `POST /oracle_router/feedback {"program_id": ..., "score": ..., "cost": ...}` from
the harness once the task is graded. `GET /oracle_router/state` (admin) shows bindings, decision-maker
state and recent feedback. The response header `x-litellm-oracle-router-model` names the bound model.

## Security

A program is private to the API key that started it: the same `program_id` under another key is another
program, so no caller can ride on, observe or complete someone else's binding. `POST /oracle_router/feedback`
completes the caller's own program; an admin key may complete any key's. The post-call hook only reads
requests the Router stamped as routed by this router (`routing_decision`), so ORACLE keys placed in the
metadata of a request to some other model are ignored, and the router drops any caller-supplied copy of
its internal metadata keys before stamping its own.

`custom` decision makers and verifiers import Python named by the deployment, so they are honoured only
for deployments in the proxy config file. A model written through the API (`/model/new`, `/model/update`,
whatever the caller's role) with a `custom` type is rejected when the router is built and never imported.

## Limits

State is in memory per proxy process; it is not persisted or shared across replicas. The paper's
dispatch scheduler (DISC) is not part of this strategy.
