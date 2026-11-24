## Scope

- This crate is the typed data model of a config and reads no file. Reading files and `include` live in `config-load`, which hands the merged document to `Config::from_value`

## Redis settings

- This crate only deserializes YAML. `RedisSettings` is the raw `coordination_redis` or `cache_params` map and `CacheSettings` tells a `type: redis` cache apart from the rest. Reading those maps the way `litellm/_redis.py` does and turning the merged kwargs into a `Resolution` both belong to `redis-resolve`

## Route settings

- `RouteList` is the typed form of a route list in config and key metadata, and both hosts read through it. A value that is not a list of strings is `RouteList::Malformed`, which grants nothing. A route template inside such a list is starlette syntax, so it is parsed by `litellm-starlette`, never here

## Environment

- `Environment` is the settings only the environment holds, the variables the proxy reads that no config.yaml key sets, one field per variable, named as the variable in lower case. The host deserializes it once and hands it to whatever builds settings from it
- `Environment::read` asks a `Lookup` for each variable by its field's name. Use it for an environment that cannot be listed, as a config's `environment_variables` over a secret manager is
- A setting config.yaml can also set is never a field here. It is a field of its config section, and `Config::with_environment` in `layer.rs` fills it when the config leaves it unset, so nothing past this crate asks which of the two set it. The rule is the same for every such key: the config's value, then `LITELLM_<KEY>`, then the default. `config.origins` keeps which keys the environment set and which variables hold text their key is not read from, for a log or an error that has to name the variable
- A flag config.yaml or a variable can set is read by the one flag rule, `lax_bool`, whichever of the two wrote it, through `LaxBool` on the field and `lax_bool` on the variable's text. It leaves this crate as an `Option<bool>`, never as `Spelled<bool>` text for a consumer to read again. A word that is no flag leaves the key unset
- The layer does not copy three things the proxy does. It reads the variable from the `Lookup` it is given, which `config-load` makes the config's `environment_variables` over the process environment, where the proxy reads some of these only at import. It reads every flag variable by the one flag rule, where the proxy has a rule per variable and one of them takes any word as on. And `reasoning_auto_summary` is config first like the rest, where the proxy turns it on from its variable even under a config that sets it to `false`
- Values stay as written: a comma-separated list is a `String`, an optional number is an `Option`. Splitting a list, applying a default and validating belong to the consumer
- `Flag` is the one parsed shape of an `Environment` field, because every consumer would otherwise repeat Python's `strip().lower() == "true"`. Python string semantics it needs come from `python-compat`. It predates the one flag rule, so do not use it for a new variable

## Database settings

- `db::settings::overlay` layers the `LiteLLM_Config` settings rows over the config document after includes and before `os.environ/` resolution. It takes rows the caller already decrypted and does no I/O. Stored `environment_variables` go into the document's own `environment_variables`, never the process environment
- `generated/db_settings.json` is script output pinning the overlay and its key tables against the Python proxy. Regenerate it with `scripts/generate_db_settings_fixtures.py` when the proxy's merge rules or `ConfigGeneralSettings` change, never by hand
