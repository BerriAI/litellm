# Authz

Whether an authenticated caller may do what its request asks: call the route, call the model. Design, order and the layers still to come live in the Notion page "Auth High-level Picture", workstream 2

These are target invariants. Where the code still lags them, the Notion page lists the gap, and new code follows the invariant, not the gap

## Boundary

- Authz decides from who the caller is and what it asks for. It never reads a counter. Budgets, rate limits and capacity are callbacks
- Every function here is pure and takes no host. Nothing here writes
- One rule per decision. A Python function that decides the same thing becomes a wrapper over it, and its body is deleted in the PR that switches it
- Records, not rows. Authz reads core-owned values and never names the `Rows` parameter. No accessor trait is added for it
- One caller type. `Caller` is the only thing that says who a request is for. A new consumer reads `Caller` or a projection of it and never adds a caller-shaped type of its own
- A record or a rule lands with the PR that first reads it. A `Layer` variant, a `ModelList` reading or a denial with no caller waits for the PR that adds one

## Denials

- Every denial here is an `AuthzDenial` in `error.rs`, and `error/AGENTS.md` holds its rules. No file under `authz/` declares an error type
- A model list's refusal names the list through `DeniedList`, so a new `Layer` adds a case there

## Model access

- `models::authorize` is the one model rule. A host supplies each list it read as a `Layer` named by the row that holds it, with the `ModelAsk` it resolved the name to, and decides nothing itself. A request has to be reached by every layer, and the first that does not names the denial. No layer, as the master key has, reaches every model
- A host that did not read a layer a caller has must have declined that caller at authentication, so no decision is ever made on a layer that is missing
- An entry is read through `litellm_router_types::ModelEntry`, which the router reads too, and a list once, in `models/list.rs`, into `ModelList`. No layer, host or router compares an entry with a special spelling itself
  - Empty, `*` or `all-proxy-models` reaches every model
  - An entry the rule does not read yet (`all-team-models`, `no-default-models`, a provider pattern such as `anthropic/*`) makes the list `Unread`, which reaches nothing. A host declines a caller with one at authentication. Reading one means porting it here, never matching it in a host
- `models/aliases.rs` gives the candidate names in the order `_can_object_call_model` applies aliases, and every layer is judged against the same set
