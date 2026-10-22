## Docs

- https://github.com/BerriAI/litellm-docs/blob/main/docs/proxy/config_settings.md

## Python

- `ConfigGeneralSettings` in `litellm/proxy/_types.py` is the Python model but is incomplete. What the proxy reads is `general_settings.get(...)`, mostly in `proxy_server.py`
- `general_settings` is one of the sections the proxy persists to the DB (`_CONFIG_PERSISTED_SECTIONS`) and the Admin UI edits, so at runtime the YAML is only one source of the effective values. Merging the DB overlay is the consumer's job, not this crate's

## Fields that need care

- `master_key`, `database_url` and `drain_endpoint_token` fall back to `LITELLM_MASTER_KEY`, `DATABASE_URL` and `DRAIN_ENDPOINT_TOKEN`. That fallback belongs to the consumer, so the fields stay optional here
- `custom_auth`, `custom_key_generate`, `custom_key_update`, `custom_key_policy`, `custom_team_metadata_validate`, `custom_sso`, `custom_ui_sso_sign_in_handler`, `litellm_jwtauth.custom_validate` and `pass_through_endpoints[].target` are Python module paths that the proxy imports. Type them as plain strings and never resolve or validate them here
- `key_management_system` and `key_management_settings` configure the secret manager itself, so they can only use `os.environ/` references, not secret manager ones
