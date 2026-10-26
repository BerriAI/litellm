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
- `ProcessEnvironment` is private to `python-bridge`, so a library constructor takes the `Lookup` as a required argument and never falls back to the process through `new()`, `Default` or an optional builder step
- An SDK that reads the process environment itself, like `aws-config`, gets every value that has an environment form from the `Lookup` first. Only sources with no environment form (profiles, ECS, IMDS) are left to the SDK

## Hand-written constructors list every field

A hand-written constructor names every field of the partial and writes `None` where its source has no such key, never `..Default::default()`. Adding a field then fails to compile in each constructor until someone decides what that source does with it

## Conversion and blank values

- String and path fields preserve their environment input, including blanks
- Numeric fields use lenient parsing and invalid values fall through
- Boolean fields reuse `core-utils::coercion::parse_str_bool`, including explicit false
- Use `read = coercion::enabled` when false must fall through, as for Python HTTP2
- Use `skip_blank` only for sources whose existing contract treats blanks as unset
- Shared conversions live in `core-utils::coercion`, not `layered`
- Use named coercions for exceptional conversion, units, or strict validation
- Keep Python SSL environment values above globals even when blank
- `label` controls display text and `litellm_setting` supplies source metadata, not a generated global reader
