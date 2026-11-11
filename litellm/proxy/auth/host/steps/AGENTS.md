# Auth host steps

Python-only work `PythonAuthHost` runs behind one `AuthHost` method, the way `litellm/router_rust/steps/` holds the router core's Python-only steps. `litellm-rust/crates/gateway-auth-core/AGENTS.md` holds the invariants, and they apply here unchanged

- Only `litellm/proxy/auth/host/python.py` imports from here, except the `Authorization` input that `auth_checks.common_checks` builds and `core.authorize` drives. Plain reads and writes of one row stay in `python.py`. A module here holds a step the core cannot decide yet because its state or its writer is Python-owned
  - `authorization.py`: the `common_checks` steps the core's authorization stage still asks for, the spend reads and budget alerts its budget checks ask for, and the `Authorization` input `auth_checks.common_checks` builds for `core.authorize`
  - `budgets.py`: the budget checks and the proxy budget alert, which read Python-owned spend counters and send alerts
  - `jwt_auto_register.py`: minting or reusing a key for a JWT claim and inserting its mapping, with the race handling for two requests registering one claim
- Each module's docstring has the same shape, so the path out of Python reads the same everywhere
  - one paragraph on what the step does
  - `TODO: moves to <Rust | gateway-auth-core> <in one piece | one check at a time>, <together with what>.`
  - `Blocked on:` bullets naming the Python-owned state or writer the core is waiting on, nested for detail
  - `After the move:` bullets saying what moves with it and what gets deleted or narrowed
- When the core decides a step itself, its module here, its `RustAuthStep` and its `answer.py` case are deleted. Its `PythonAuthHost` method is deleted too, unless the core still needs the host to read Python-owned state or send an alert, in which case it narrows to that read or hostcall
