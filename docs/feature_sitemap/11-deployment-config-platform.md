# 11 Deployment, config, and platform

Running the gateway itself: config.yaml loading and overrides, the CLIs, Docker/helm, the database and Redis layers, secret managers, multi-region, hooks and plugins, request limits, server tuning, UI customization, the license gate, and upgrades. Feature-specific config keys are filed under their own domains; this file covers the platform surface they all stand on

## Config and deployment

### platform.config_yaml: config.yaml loading, environment_variables, os.environ/ refs, include
surfaces: config, cli | flags: none
docs: https://docs.litellm.ai/docs/proxy/configs, https://docs.litellm.ai/docs/proxy/config_settings
code: `litellm/proxy/proxy_server.py` (config load path), `litellm/proxy/_types.py` (`ConfigYAML`), `litellm/proxy/proxy_cli.py` (`--config` flag)
tests: `tests/test_litellm/proxy/` (config load files)
registry: other.yaml other.config.*
verify: boot with `python litellm/proxy/proxy_cli.py --config config.yaml`, reference a secret as `os.environ/NAME`, and confirm it resolves from env

### platform.config_management: Config in DB, config reload, /config/* endpoints, config_overrides
surfaces: api, config, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/config_management
code: `litellm/proxy/proxy_server.py` (`/config/update`, `/config/field/*`, `/config/list`, `/config/yaml`), `litellm/proxy/management_endpoints/config_override_endpoints.py` (`/config_overrides/*`)
tests: `tests/test_litellm/proxy/` (config management files)
registry: other.yaml other.config.*
verify: POST /config/field/update to change a general_settings key, then GET /config/field/info to read it back without a restart

### platform.cli: litellm CLI (start proxy, flags)
surfaces: cli | flags: none
docs: https://docs.litellm.ai/docs/proxy/cli, https://docs.litellm.ai/docs/proxy/quick_start
code: `litellm/proxy/proxy_cli.py` (`litellm` entrypoint)
tests: `tests/test_litellm/` (cli related files)
registry: none
verify: run `litellm --model gpt-5.5` and call http://localhost:4000/chat/completions without any config file

### platform.management_cli: litellm-proxy management CLI
surfaces: cli | flags: none
docs: https://docs.litellm.ai/docs/proxy/management_cli
code: `litellm/proxy/` (management cli entrypoint, check `pyproject.toml` scripts)
tests: `tests/test_litellm/` (management cli files)
registry: none
verify: run the management cli against the proxy to create a key per the doc and confirm it shows in /key/list

### platform.docker_helm: Docker images, helm chart, hardened image, render/railway
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/deploy, https://docs.litellm.ai/docs/proxy/docker_quick_start, https://docs.litellm.ai/docs/proxy/docker_image_security
code: `Dockerfile`, `helm/`, `docker-compose.yml`
tests: `tests/e2e/migrations/` (container boot coverage)
registry: other.yaml other.lifecycle.*
verify: `docker run` the published image with a mounted config and master key, then hit /health/liveliness

### platform.database: Postgres, Prisma migrations, connection pool settings, read replica, db_scripts
surfaces: config | flags: db
docs: https://docs.litellm.ai/docs/proxy/db_info, https://docs.litellm.ai/docs/proxy/db_sizing, https://docs.litellm.ai/docs/proxy/db_read_replica, https://docs.litellm.ai/docs/troubleshoot/prisma_migrations
code: `schema.prisma`, `migrations/`, `db_scripts/`, `litellm/proxy/db/` if present
tests: `tests/e2e/migrations/`
registry: none
verify: point DATABASE_URL at Postgres, boot, and confirm prisma migrate runs and /key/generate persists across restart

### platform.redis: Redis requirements, cluster, sentinel, coordination redis
surfaces: config | flags: redis
docs: https://docs.litellm.ai/docs/proxy/redis_requirements, https://docs.litellm.ai/docs/proxy/redis_sizing
code: `litellm/_redis.py` (client init), `litellm/caching/` (redis backends), `litellm/proxy/` (redis coordination uses)
tests: `tests/test_litellm/caching/` (redis files)
registry: reliability.yaml reliability.circuit_breaker.redis.* (redis-backed behavior)
verify: set `REDIS_HOST`/`REDIS_PORT`, boot two proxy replicas against it, and confirm cooldown/rate-limit state is shared

### platform.secret_managers: Secret managers (AWS, GCP, Azure Key Vault, Hashicorp, CyberArk, custom)
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/secret_managers/overview, https://docs.litellm.ai/docs/secret_managers/aws_secret_manager, https://docs.litellm.ai/docs/secret_managers/hashicorp_vault, https://docs.litellm.ai/docs/secret_managers/cyberark, https://docs.litellm.ai/docs/secret_managers/custom_secret_manager
code: `litellm/secret_managers/`, `litellm/secret_managers/aws_secret_manager.py`, `litellm/secret_managers/hashicorp_secret_manager.py`, `litellm/secret_managers/cyberark_secret_manager.py`, `litellm/secret_managers/custom_secret_manager_loader.py`, `litellm/integrations/custom_secret_manager.py`
tests: `tests/test_litellm/` (secret manager files), `tests/e2e/secret_manager/` (real Vault/CyberArk lanes)
registry: other.yaml other.config.secret_manager.*, other.config.secret_resolution.*
verify: set `general_settings.key_management_system: hashicorp_vault` with vault env, store a key in vault, and confirm a deployment resolves `os.environ/NAME` from it

### platform.key_management_system: Virtual keys stored in secret manager
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/secret_managers/overview
code: `litellm/secret_managers/secret_manager_handler.py`, `litellm/proxy/management_endpoints/key_management_endpoints.py` (key write-through)
tests: `tests/e2e/secret_manager/` (virtual key write/delete coverage)
registry: other.yaml other.key_mgmt.*
verify: with key_management_system enabled, POST /key/generate, then confirm the key secret exists in the manager and is deleted on /key/delete

### platform.multi_region_control_plane: Global control plane and multi-region
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/proxy/global_control_plane, https://docs.litellm.ai/docs/proxy/multi_region, https://docs.litellm.ai/docs/proxy/multi_tenant_architecture
code: `litellm/proxy/` (region-aware config and sync), `litellm/_redis.py` (coordination)
tests: `tests/test_litellm/proxy/` (multi region files)
registry: none
verify: deploy a second region pointing at the shared DB/redis per the doc and confirm keys created in region A work in region B

## Hooks, plugins, limits, tuning

### platform.worker_startup_hooks: Worker startup hooks, plugins (general_settings.plugins)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/worker_startup_hooks, https://docs.litellm.ai/docs/proxy/plugins
code: `litellm/proxy/plugin_routes.py` (`/api/plugins`, `/plugin-proxy/{plugin_name}/{path:path}`), `litellm/proxy/proxy_server.py` (startup hook dispatch)
tests: `tests/test_litellm/proxy/` (plugin files)
registry: none
verify: register a plugin in general_settings.plugins, boot, and confirm GET /api/plugins lists it and its routes answer

### platform.custom_hooks: Custom call hooks (CustomLogger async_pre_call_hook, post_call rules)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/call_hooks, https://docs.litellm.ai/docs/proxy/rules, https://docs.litellm.ai/docs/observability/custom_callback
code: `litellm/integrations/custom_logger.py` (`CustomLogger` hook methods), `litellm/proxy/utils.py` if present (hook dispatch), `litellm/proxy/hooks/`
tests: `tests/test_litellm/integrations/` (custom logger files)
registry: none
verify: write a CustomLogger subclass whose async_pre_call_hook raises on a marker string, register it via custom_callbacks, and confirm the block

### platform.request_limits: Request/response size limits, file extension allow/block lists
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/guides/security_settings
code: `litellm/proxy/proxy_server.py` (request size checks), `litellm/proxy/_types.py` (size/extension settings), `litellm/proxy/middleware/` if present
tests: `tests/test_litellm/proxy/` (request limit files)
registry: none
verify: set a max request size per the doc, send an oversized body, and confirm a 413-style rejection

### platform.performance: Server tuning, high throughput, num_workers
surfaces: config, cli | flags: none
docs: https://docs.litellm.ai/docs/proxy/server_tuning, https://docs.litellm.ai/docs/proxy/high_throughput, https://docs.litellm.ai/docs/proxy/prod, https://docs.litellm.ai/docs/load_test_rpm
code: `litellm/proxy/proxy_cli.py` (`--num_workers`, uvicorn flags), `litellm/proxy/proxy_server.py`
tests: `tests/e2e/load/` (load suites, opt-in)
registry: none
verify: boot with `--num_workers 4`, run the rpm load recipe from the docs, and confirm throughput scales vs a single worker

### platform.ui_customization: UI logo, theme, custom root, docs URL
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/custom_root_ui, https://docs.litellm.ai/docs/proxy/ui/ui_edit_logo
code: `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/ui_theme_settings`, `/upload/logo`), `litellm/proxy/proxy_server.py` (`/get_logo_url`, `/get_favicon`, `/get_image`)
tests: `tests/test_litellm/proxy/` (ui customization files)
registry: none
verify: POST /upload/logo with an image, reload the UI, and confirm the logo and theme settings render

### platform.admin_settings_ui: Admin settings page
surfaces: ui, api | flags: db
docs: https://docs.litellm.ai/docs/proxy/ui
code: `ui/litellm-dashboard/src/app/(dashboard)/admin-panel/page.tsx`, `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/ui_settings`, `/update/ui_settings`)
tests: `tests/test_litellm/proxy/`
registry: none
verify: open http://localhost:4000/ui/?page=admin-panel, change a setting, and confirm it persists via the corresponding GET

### platform.api_reference_ui: Swagger / API reference
surfaces: ui, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/architecture (API surface overview)
code: `litellm/proxy/proxy_server.py` (FastAPI `/docs`, `_get_openapi_url`), `ui/litellm-dashboard/src/app/(dashboard)/api-reference/page.tsx`
tests: `tests/test_litellm/proxy/`
registry: none
verify: open http://localhost:4000/docs (Swagger) and http://localhost:4000/ui/?page=api-reference in the Admin UI

### platform.license: Enterprise license and premium gating
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/proxy/architecture (licensing), https://docs.litellm.ai/docs/shared_responsibility
code: `litellm/proxy/auth/` (license check), `litellm/proxy/proxy_server.py` (`premium_user` gating), `litellm/proxy/health_endpoints/_health_endpoints.py` (GET `/health/license`)
tests: `tests/test_litellm/proxy/` (license files)
registry: none
verify: call an enterprise endpoint without LITELLM_LICENSE set and confirm the not_premium_user error, then set it and retry

### platform.rollback_upgrade: Version rollback and upgrade paths
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/migration, https://docs.litellm.ai/docs/migration_policy, https://docs.litellm.ai/docs/troubleshoot/rollback, https://docs.litellm.ai/docs/troubleshoot/pip_venv_upgrade, https://docs.litellm.ai/docs/proxy/release_cycle, https://docs.litellm.ai/docs/api_stability_policy
code: `migrations/` (prisma migrations), `litellm/proxy/` (version gates)
tests: `tests/e2e/migrations/` (legacy DB compat)
registry: none
verify: boot a new version against a DB written by the previous release and confirm migrations apply and the proxy serves traffic
