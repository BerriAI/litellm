## Scope

- This crate is the gateway's config loader. It takes the types `litellm-config` parses and turns them into the values the gateway runs on, so env lookups, the DB overlay, reference substitution, defaults and the domain types live here, never in `litellm-config`
- It never reads process state. The caller reads the process environment once and passes it in as an `Environment`, so tests pass their own variables
- Python never uses this crate. It shares only the `litellm-config` types, so behavior here may differ from Python where Python is wrong, and each difference is listed in the config compat doc

- Fixture and corpus rules live in `tests/AGENTS.md`

## Precedence

- Each lowered value takes the env, the file and then the default in the order Python uses for that key today, until that order is normalized
