# litellm/_auth

## Why this exists

- Lots of Python code like `litellm/proxy/auth/user_api_key_auth.py` is too complex
- Soon, Rust will implement the same auth stack, and expose common functionality as Python API through pyo3, to avoid maintaining 2 copy of code.
- So the goal of `_auth` here is to define that interface first, not writing really good python code. Should be behavior preserving refactor though

## The swap point

- `decision.py` is the interface. `AuthCore.decide(RequestFacts, Settings) -> Outcome` is the only thing a core implements, and `python_core.PythonCore` is the Python one. A Rust core will implement the same protocol through pyo3, and the builder picks one without changing anything else
- A core does no IO and reads no globals. The same facts and settings always give the same outcome
- `Defer` means the core does not handle this request yet, and the legacy builder in `litellm/proxy/auth/user_api_key_auth.py` takes it from there. Move one decision at a time out of the builder into the core, and delete the builder code it replaces in the same change
- Projection stays in the proxy. `Allow` grants become `UserAPIKeyAuth` in the builder, and `litellm/proxy/auth/auth_denial.py` maps every `DenyReason` to the existing error contract
- When a decision needs a lookup, add it as an effect the core yields and the proxy answers, so both cores share the same IO code
