# Online Model Experiments

## Decision

Add a fixed-variant online experiment capability for comparing models on live
traffic. An experiment assigns a stable subject to one model variant, records
the assignment with every request, and exposes aggregated operational metrics
for comparison.

This is distinct from the existing adaptive router. The adaptive router makes
per-turn decisions based on learned signals. An online experiment must preserve
the assignment long enough to make a fair comparison and must not change the
traffic split because of early observations.

## Problem

Operators can route traffic across model groups and inspect usage, cost, and
latency, but they do not have a first-class way to answer which model performs
better for a real workload. A useful comparison needs the same population,
stable assignment, model-level metrics, and a clear indication of sample size.

## MVP scope

The first version should support:

- one experiment with two or more model variants;
- percentage-based traffic allocation;
- active, paused, completed, and cancelled states;
- stable assignment by end user, session, API key, team, or project;
- model, variant, experiment, and assignment metadata on each request;
- cost, token, latency, retry, fallback, and error aggregation;
- explicit end-user feedback with a positive, negative, or unknown value;
- an authenticated read-only summary endpoint for the dashboard;
- a dashboard table and time-series charts;
- no automatic promotion in the MVP.

The MVP should not add an LLM judge, statistical significance claims, automatic
rollouts, or a second storage system for request logs.

## Assignment contract

Assignment must be deterministic for the lifetime of the experiment. The
assignment key should be a server-derived stable identity, in this order:

1. authenticated end-user identity when available;
2. session identity when supplied by the caller;
3. API key, team, or project identity as an explicit fallback.

The implementation must hash the identity with an experiment-specific salt. Raw
credentials and personally identifying values must never be persisted as the
assignment key.

The selected variant should be computed from a deterministic hash bucket rather
than a random choice on each request. Changing the allocation of a running
experiment must be an explicit versioned change, otherwise existing subjects
could silently move between variants.

## High-level flow

```text
Request
  -> authenticate and resolve experiment scope
  -> derive privacy-safe assignment identity
  -> deterministically select variant
  -> route through the existing model group machinery
  -> attach experiment metadata to the request and log record
  -> aggregate existing spend, latency, error, and feedback signals
  -> expose summary data to the dashboard
```

The experiment layer should choose a logical model group. Existing router
selection, provider fallback, budgets, guardrails, and spend tracking should
remain responsible for executing the request.

## Proposed entities

`ModelExperiment` should contain the experiment name, description, status,
assignment scope, start and end timestamps, and the current allocation
revision.

`ModelExperimentVariant` should contain the experiment id, variant name, target
model group, allocation percentage, and display order.

`ModelExperimentAssignment` should contain the experiment id, hashed assignment
key, variant id, allocation revision, and timestamps. The experiment and hashed
assignment key should be unique together.

Request logs should carry the experiment id, variant id, and allocation revision
through the existing metadata path. Aggregates can initially be calculated from
existing logs and spend tables, with a later rollup added only if query cost
requires it.

## API shape

The initial API can follow existing management endpoint conventions:

```text
POST   /model-experiments
GET    /model-experiments
GET    /model-experiments/{experiment_id}
PATCH  /model-experiments/{experiment_id}
POST   /model-experiments/{experiment_id}/pause
POST   /model-experiments/{experiment_id}/complete
GET    /model-experiments/{experiment_id}/metrics
```

The metrics response should return both aggregates and sample counts. Every
metric needs a denominator. For example, error rate without request count is
not actionable, and a quality score without evaluated responses is misleading.

## Metrics

The first dashboard should show:

| Metric | Definition |
| --- | --- |
| Requests | Completed requests attributed to the variant |
| Subjects | Unique assigned identities |
| Cost | Total and average spend |
| Latency | Average, p50, p95, and p99 when available |
| Errors | Failed requests divided by requests |
| Retries | Requests requiring a retry |
| Fallbacks | Requests served by a fallback model |
| Feedback | Positive, negative, and unrated feedback |
| Quality proxy | Positive feedback divided by rated feedback |

The UI should label quality as a proxy until the project has a stronger
evaluation contract. It should also show a small-sample warning rather than
declaring a winner from a tiny number of observations.

## Safety and fairness

The system must keep the same user on the same variant for conversational
workloads. It must preserve guardrails and authorization before assignment and
must not allow an experiment to bypass model access rules, budgets, or provider
restrictions.

Metrics must be filterable by team, project, model, and time window. The API
must not expose raw prompts or sensitive response content as part of the
experiment summary.

## Implementation sequence

1. Define typed experiment and variant contracts plus deterministic assignment.
2. Add unit tests for allocation, stickiness, paused experiments, invalid
   allocations, and assignment revision changes.
3. Integrate selection before the existing model-group routing path.
4. Add experiment metadata to the existing logging and cost path.
5. Add a read-only metrics endpoint backed by existing data.
6. Add a minimal dashboard page with the comparison table and charts.
7. Validate with a local proxy using two controlled upstreams before proposing
   the feature publicly.

## Revisit as the feature grows

Later versions can add canary ramps, statistical confidence intervals, quality
judges, human review queues, automatic rollback, and promotion into a model
group. Those features should be layered on top of the fixed assignment and
evidence model rather than mixed into the first routing implementation.
