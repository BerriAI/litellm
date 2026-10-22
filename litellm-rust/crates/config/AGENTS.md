## Scope

- The crate is the type definition of `config.yaml`, what a user writes and nothing more. Parsing one YAML text into those types (`Config::from_yaml`, `Config::from_mapping`, the PyYAML reading rules in `yaml.rs`) is all it does
- Everything that turns those types into what the gateway runs on lives in `crates/config-load`: resolving `include`, overlaying DB rows, reading the environment, substituting references, applying defaults and building the domain types. So this crate never reads files, the environment or a secret manager
- It knows nothing about which settings a consumer implements or rolls out, so checks like "refuse setting X because it is not supported yet" live in the consuming crate (e.g. `gateway`), never here

## Defaults

- The value the existing proxy uses when a key is absent (`health_check_interval` is 300, `database_connection_pool_limit` is 10) is part of what `config.yaml` means, but it is applied in `crates/config-load`, never here. Parsing never fills a default in, so every key here keeps whether the file set it. The proxy treats a key set in the file as owned by the file, so the DB value and Admin UI edits for it are ignored
- Other sections have no DB overlay, so a key there whose default is empty (an empty list or map) can be the plain collection, filled in by `#[serde(default)]`

## Module layout

Each top-level `config.yaml` key belongs to exactly one module under `src/`. A module is a single `<name>.rs` file until it has flattened sub-groups, then it becomes a `<name>/` folder with one file per sub-group. This only applies one level down, so groups flattened inside a sub-group (like the cache backends in `litellm/caching.rs`) stay in that sub-group's file. Keys typed as a plain `Object` have no module and stay on `Config` in `lib.rs`. Keep this list in sync when adding a key

- `model.rs`: `model_list`
- `general/`: `general_settings`
- `router.rs`: `router_settings`
- `litellm/`: `litellm_settings`
- `callback.rs`: `callback_settings`
- `provider.rs`: `assistant_settings`, `default_vertex_config`, `files_settings`, `finetune_settings`, `worker_registry`
- `mcp.rs`: `mcp_servers`
- `credential.rs`: `credential_list`
- `guardrail.rs`: `guardrails`
- `prompt.rs`: `prompts`
- `tool.rs`: `sandbox_tools`, `search_tools`
- `vector_store.rs`: `vector_store_registry`
- `agent.rs`: `agents`, `agent_list`
- `policy.rs`: `policies`, `policy_attachments`
- `lib.rs`: `environment_variables`, `mcp_tools`, `include`

`error.rs`, `value.rs` and `yaml.rs` are crate infrastructure, not config keys

## Serde flatten

Sections split one YAML mapping into several structs with nested `#[serde(flatten)]` (`LitellmSettings` flattens `CachingSettings`, `GeneralSettings` flattens `AuthSettings`, and so on). Serde hands keys to flattened fields like this

- A flattened struct without a flatten field of its own takes only the keys it declares, and later siblings and the catch-all never see them
- A flattened struct or map that has its own flatten field is read as a map and claims nothing. It sees every key still unclaimed, including keys a later sibling declares, and those keys also reach the outer catch-all
- Fields claim keys in declaration order, so when two flattened structs declare the same key only the first one gets it

Which gives these rules

- Each mapping has exactly one `#[serde(flatten)] additional_fields: Object`, declared last on the outermost struct of that mapping. Flattened sub-structs stay plain, with no flatten field inside
- A key belongs to exactly one struct. Which one is decided by who reads it, see `src/AGENTS.md`
- Never combine `deny_unknown_fields` with flatten, on either the outer or the flattened struct. It is fine on a struct that is never flattened and has no flatten fields

## Inline YAML

- Write inline YAML as a multi-line raw string (`r#"..."#`), not as a one-line string with `\n` escapes like `"key:\n  value"`
- This includes `#[case(...)]` arguments. Open the raw string on the attribute line and start the YAML at column 0, e.g. `#[case::name(r#"` then the YAML lines then `"#)]`
- YAML that fits on one line, like flow-style `key: {a: 1}`, stays a plain string with no trailing `\n`

## Debug and redaction

Every struct derives `Debug`, and redaction lives on the field types, so no struct has a hand-written `Debug`

- Anything that is or can carry a secret (keys, tokens, passwords, URLs that may embed credentials, MCP `command` and `args`) is `SecretValue`, which prints `[REDACTED]`
- YAML we don't model is `Object` (prints its keys) or `Raw` (prints its kind). They are the only hand-written `Debug` impls. Never put a bare `Value` in a field
- Redaction tests put a unique `SENTINEL` under the slot and assert it never shows up in `format!("{config:?}")`. Words like `password` also match field names such as `redis_password`
