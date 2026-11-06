# litellm/_auth

## Why this exists

- Lots of Python code like `litellm/proxy/auth/user_api_key_auth.py` is too complex
- Soon, Rust will implement the same auth stack, and expose common functionality as Python API through pyo3, to avoid maintaining 2 copy of code.
- So the goal of `_auth` here is to define that interface first, not writing really good python code. Should be behavior preserving refactor though

## The swap point

- `decision.py` is the interface. `AuthCore.start(RequestFacts, Settings, now) -> Step` is the only thing a core implements, and `python_core.PythonCore` is the Python one. A Rust core will implement the same protocol through pyo3, and the builder picks one without changing anything else
- A core does no IO and reads no globals. The same facts, settings, time and effect answers always give the same outcome
- A `Step` is either an `Outcome` or an effect. Each effect carries its typed `resume` continuation, and `driver.drive` answers effects through an `EffectHandler` until an outcome comes back. The proxy's handler (`_BuilderEffects` in `litellm/proxy/auth/user_api_key_auth.py`) owns all cache and DB IO, so both cores share it
- Projection stays in the proxy. `_authenticated` in the builder module turns an `Allow` grant into `UserAPIKeyAuth`, and `litellm/proxy/auth/auth_denial.py` maps every `Denial` to the existing error contract
- The key checks run in the core, in the order the proxy has always applied them: blocked key, model access, SCIM-deactivated owner, budget exemption, team member budget, expiry, budget alerts, the key's own max budget (or its throttle), the soft budget alert, per-model budgets, the team and project context, the proxy budget alert, then the key's token. A budget-exempt request skips everything from the team member budget to per-model budgets except expiry. `start_resolved_key` runs the same checks for keys the proxy resolved itself, such as internal jobs
- Per-model budgets are still one opaque effect (`EnforceModelBudgets`), because the key, team member and custom-auth paths share their fallback logic. Move that logic into the core for every path at once rather than forking it per path
