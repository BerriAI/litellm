# steps

Answers to router-core's host steps that need live Python objects. router-core makes the routing decision; these modules supply the Python-only facts or run the Python-only objects it asks about.

- `selection`: stateful strategies (latency, tpm/rpm, cost, usage) on `select` steps
- `prerouting`: auto, complexity, adaptive and quality routers, and session binding, on `pre_route` steps
- `precall`: token counts, context windows and param support for pre-call checks
- `plugins`: `Router(plugins=[...])` objects
- `callbacks`: the optional pre-call checks (budgets, affinity, rate limits) as `CustomLogger`s
- `access`: the caller facts read from auth metadata

A Rust host answers these steps natively, or declines the config at build.
