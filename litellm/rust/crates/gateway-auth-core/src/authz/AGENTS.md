# Authz

Whether an authenticated caller may do what its request asks: call the route, call the model. Design, order and the list of open copies live in the Notion page "Authz Design" under "Rust Migration Depth First Plan / Auth"

These are target invariants. Where the code still lags them, the Notion page lists the gap, and new code follows the invariant, not the gap

## Boundary

- Authz decides from who the caller is and what it asks for. It never reads a counter. Budgets, rate limits and capacity are callbacks, and authz only hands on the limits a caller carries
- Every function here is pure and takes no host. `authorize` for a whole request, which will read through `AuthStore` and ask `AuthHost` for Python-owned state, is the one exception, and it only reads. Nothing here writes
- One rule per decision. A Python function that decides the same thing becomes a wrapper over it, and its body is deleted in the PR that switches it. Until then the rule is listed under "Open copies" on the Notion page
- Records, not rows. Authz reads core-owned values and never names the `Rows` parameter. No accessor trait is added for it
- One caller type. `Caller` is the only thing that says who a request is for. A new consumer reads `Caller` or a projection of it and never adds a caller-shaped type of its own

## Denials

- Every denial is an `AuthzDenial` in `denial.rs`. It carries the exception class the Python proxy raises, the status, the `type` and `param`, and the message. Keep each opening sentence stable, since clients match on it
- The `type` a model denial carries is the one Python answers for that list today, uneven as it is: a team member's is the team's, and a user whose list is `no-default-models` gets the key's. Changing one is a deliberate change with its own PR
- `public.rs` stays the one mapping to `PublicDenial`. A model denial is raised as a class of its own, so mapping it needs a new `PublicDenial` variant, which the Python bridge matches exhaustively. That mapping lands with the Python switch
- Side effects are returned values, such as the matched access groups. The host stamps them once

## Model access

- `models::authorize` is the one model rule. A host supplies each list it read as a `Layer` named by the row that holds it, with the `ModelAsk` it resolved the name to, and decides nothing itself. A request has to be reached by every layer, and the first that does not names the denial. No layer, as the master key has, reaches every model
- A host that did not read a layer a caller has must have declined that caller at authentication, so no decision is ever made on a layer that is missing
- Which layers a caller has follows `common_checks`: the user's list is a layer only for a key with no team, and a project's or an end user's only when it names something
- A list's sentinels are read once, in `models/list.rs`, into `ModelList`. No layer and no host compares an entry with a sentinel itself
  - Empty, `*` or `all-proxy-models` reaches every model
  - `all-team-models` leaves the team's list to decide on the layers the proxy reads under a team: the key's, the team's own, a member's and a project's. It is never every model the proxy serves, which `check_model_access_helper` made of it. On a user's, an organization's or an end user's list it reaches nothing, as the proxy reads it there as a model's name
  - A team member with no `allowed_models` leaves the team's list to decide too
  - `no-default-models` reaches nothing
- A list reaches a model through its entries or through the models its access groups add, the same for every layer
- `models/aliases.rs` resolves the candidate names once, in the order `_can_object_call_model` applies aliases, and every layer is judged against the same set. A name the team aliases is the team's own grant, so only the key's and the team's lists reach it whatever they hold
- Not read yet, so a host declines a caller that needs one. Reading one means porting it here, never matching it in a host
  - A pattern entry such as `anthropic/*`, which the proxy matches against the model's provider
  - A key's access group that overrides the team's refusal (`key_access_group_grants_model`)
  - The names only the Python router knows: a context compaction parent and a team deployment's public name
