## Docs

- https://github.com/BerriAI/litellm-docs/blob/main/docs/proxy/config_settings.md

## Python

- There is no Python model. `ConfigYAML.litellm_settings` in `litellm/proxy/_types.py` is a plain `dict`
- `ProxyConfig.load_config` in `proxy_server.py` handles a few keys specially (`cache`, `cache_params`, `callbacks`, `success_callback`, `guardrails`, `drop_params` and others) and does `setattr(litellm, key, value)` for everything else. So any module-level attribute in `litellm/__init__.py` is a valid key, and that is the real key set
- `litellm_settings` is one of the sections the proxy persists to the DB (`_CONFIG_PERSISTED_SECTIONS`) and the Admin UI edits, so at runtime the YAML is only one source of the effective values. Merging the DB overlay is the consumer's job, not this crate's
