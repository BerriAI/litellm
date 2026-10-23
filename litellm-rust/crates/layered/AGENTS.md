# layered

`#[derive(Layered)]` declares where each setting of a group comes from. The derive generates `<Name>Partial`, with `from_env` and `from_litellm_params`, and the name lists and `Sources` rows for errors and projection. `merge` stacks partials highest first, and the group's crate lowers the merged partial into its final type, where defaults live

## Runtimes stack different sources

| Runtime | Stack, highest first | Where config.yaml shows up |
|---|---|---|
| Python SDK | litellm params, env, `litellm.*` globals, defaults | nowhere |
| Python proxy | litellm params, env, `litellm.*` globals, defaults | already copied into env and globals by the proxy |
| Rust gateway | litellm params, env, config file, defaults | read by the gateway |

Each runtime writes its own `merge([..])` list per group, and a test per runtime pins it. Credentials put litellm params first and the global moves per provider, so never assume one order across groups

## Who builds which partial

- `from_env` and `from_litellm_params` are generated, and every runtime that has that source calls them
- A source holding typed values (the config file, a live Python global) gets a hand-written constructor in the crate that owns the source, never in `layered` or in the group's crate
- `layered` never depends on `litellm-config` or pyo3, and no library crate reads the process environment itself. It takes a `Lookup`: `Env` in the Rust gateway, `ProcessEnvironment` in `python-bridge`

## Hand-written constructors list every field

A hand-written constructor names every field of the partial and writes `None` where its source has no such key, never `..Default::default()`. Adding a field then fails to compile in each constructor until someone decides what that source does with it

## Blank means unset

An empty or whitespace-only env value or string litellm param counts as unset, so the lookup moves on to the next name and the next layer. Mark a field `keep_blank` only when an empty value is a deliberate choice, like `LITELLM_USER_AGENT=""` sending no user agent. Python's `os.getenv(X, litellm.x)` lets an empty env var shadow the global for the SSL settings, and we diverge from that on purpose
