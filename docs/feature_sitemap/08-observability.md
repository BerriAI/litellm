# 08 Observability

What the gateway emits about traffic: the callback framework and logging integrations, OpenTelemetry and Prometheus, alerting and email, the Logs UI page, redaction, audit logs, and health/debug endpoints. Spend reporting lives in [05-budgets-ratelimits-spend.md](05-budgets-ratelimits-spend.md); the callback-by-callback list is in [appendix-integrations.md](appendix-integrations.md)

## Logging pipeline

### obs.callbacks: Logging callbacks framework (success/failure/service callbacks, CustomLogger, standard_logging_payload)
surfaces: sdk, config | flags: none
docs: https://docs.litellm.ai/docs/observability/callbacks, https://docs.litellm.ai/docs/proxy/logging_spec, https://docs.litellm.ai/docs/proxy/logging
code: `litellm/litellm_core_utils/litellm_logging.py`, `litellm/integrations/custom_logger.py` (`CustomLogger`), `litellm/litellm_core_utils/custom_logger_registry.py` (`CustomLoggerRegistry`)
tests: `tests/test_litellm/litellm_core_utils/` (logging files), `tests/test_litellm/integrations/`
registry: logging.yaml logging.* (delivery cases per integration)
verify: add `litellm_settings.callbacks: ["langfuse"]` with LANGFUSE keys, run a call, and see the trace in Langfuse

### obs.integrations: Logging integrations (langfuse, datadog, otel, s3, gcs, langsmith, arize, mlflow, ... 50+)
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/observability/langfuse_integration, https://docs.litellm.ai/docs/observability/datadog, https://docs.litellm.ai/docs/observability/langsmith_integration
code: `litellm/integrations/` (one module/dir per integration), `litellm/litellm_core_utils/custom_logger_registry.py`
tests: `tests/test_litellm/integrations/`, `tests/e2e/logging/`
registry: logging.yaml logging.<name>.*
verify: enable one integration (e.g. datadog with DD_API_KEY/DD_SITE), run a call, and confirm a log row arrives; full table in [appendix-integrations.md](appendix-integrations.md)

### obs.otel: OpenTelemetry tracing and metrics
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/observability/opentelemetry_integration, https://docs.litellm.ai/docs/observability/opentelemetry_v2, https://docs.litellm.ai/docs/observability/opentelemetry_v2_migration
code: `litellm/integrations/opentelemetry.py`, `litellm/integrations/otel/`, `litellm/proxy/anthropic_endpoints/gateway_endpoints.py` (POST `/v1/traces`, `/v1/metrics`, `/v1/logs` ingestion)
tests: `tests/test_litellm/integrations/open_telemetry/`
registry: logging.yaml logging.otel.*
verify: set `litellm_settings.callbacks: ["otel"]` with OTEL_EXPORTER_OTLP_ENDPOINT, run a call, and see the span in your collector

### obs.prometheus: Prometheus metrics and /metrics
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/proxy/prometheus, https://docs.litellm.ai/docs/proxy/billing_metrics
code: `litellm/integrations/prometheus.py`, `litellm/integrations/prometheus_services.py`, `litellm/proxy/prometheus_metrics_server.py` (`/metrics`)
tests: `tests/test_litellm/integrations/` (prometheus files)
registry: logging.yaml logging.prometheus.*
verify: set `litellm_settings.callbacks: ["prometheus"]`, run calls, and curl http://localhost:4000/metrics for litellm_* series

### obs.dynamic_logging: Dynamic per-request / per-key / per-team logging
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/dynamic_logging
code: `litellm/proxy/litellm_pre_call_utils.py` (dynamic callback fields), `litellm/integrations/custom_logger.py`
tests: `tests/test_litellm/proxy/` (dynamic logging files)
registry: none
verify: pass `metadata: {"callbacks": ["langfuse"]}` (per the doc's exact field) on one request and confirm only that request logs to the added sink

### obs.team_logging: Team and key logging settings
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/team_logging
code: `litellm/proxy/management_endpoints/team_callback_endpoints.py` (`/team/{team_id}/callback`), `litellm/proxy/management_endpoints/team_endpoints.py` (logging fields on teams)
tests: `tests/test_litellm/proxy/management_endpoints/` (team callback files)
registry: mgmt.yaml mgmt.callback.*
verify: POST /team/{team_id}/callback with a langfuse callback, run a team-key call, and confirm it lands in that team's project only

### obs.logging_settings_ui: Logging and alerts settings page
surfaces: ui, api | flags: db
docs: https://docs.litellm.ai/docs/proxy/logging
code: `litellm/proxy/management_endpoints/callback_management_endpoints.py` (`/callbacks/list`, `/callbacks/configs`), `ui/litellm-dashboard/src/app/(dashboard)/logging-and-alerts/page.tsx`
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.callback.*
verify: open http://localhost:4000/ui/?page=logging-and-alerts, add a callback, and confirm it shows in GET /callbacks/list

### obs.alerting: Alerting (Slack, MS Teams, webhook, PagerDuty; alert types)
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/alerting, https://docs.litellm.ai/docs/proxy/pagerduty, https://docs.litellm.ai/docs/observability/slack_integration
code: `litellm/integrations/SlackAlerting/`, `litellm/integrations/email_alerting.py`, `litellm/proxy/proxy_server.py` (GET `/alerting/settings`)
tests: `tests/test_litellm/integrations/` (alerting files)
registry: none
verify: set SLACK_WEBHOOK_URL and `alerting: ["slack"]` with an alert_type, trigger a budget-crossing call, and confirm the Slack message

### obs.email: Email events (invitations, budget alerts; SMTP, Resend, SendGrid)
surfaces: config, ui | flags: ent
docs: https://docs.litellm.ai/docs/proxy/email
code: `enterprise/litellm_enterprise/enterprise_callbacks/send_emails/` (`/email/event_settings`), `litellm/integrations/email_templates/`
tests: `tests/test_litellm/` (email related files)
registry: none
verify: configure SMTP in general_settings, POST /invitation/new, and confirm the invitation email is sent

## UI logs and redaction

### obs.logs_page: Logs page (request logs, sessions, error logs)
surfaces: ui, api | flags: db
docs: https://docs.litellm.ai/docs/proxy/ui_logs, https://docs.litellm.ai/docs/proxy/ui_logs_sessions
code: `litellm/proxy/spend_tracking/spend_management_endpoints.py` (GET `/spend/logs/ui`, `/spend/logs/session/ui`, `/spend/logs/ui/{request_id}`), `ui/litellm-dashboard/src/app/(dashboard)/logs/page.tsx`
tests: `tests/e2e/management/`
registry: quota_management.yaml quota_management.spend_tracking.*
verify: run a few calls, open http://localhost:4000/ui/?page=logs, and click a row to see request detail

### obs.spend_log_settings: Spend log settings (redaction, prompt storage)
surfaces: ui, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/ui_spend_log_settings
code: `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/ui_settings`, `/update/ui_settings`), `litellm/proxy/spend_tracking/` (prompt storage settings)
tests: `tests/test_litellm/proxy/`
registry: none
verify: toggle `store_prompts_in_spend_logs` in the UI settings, run a call, and confirm the prompt column in /spend/logs is empty

### obs.redaction: Message and key redaction in logs (turn_off_message_logging, redact_user_api_key_info)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/logging, https://docs.litellm.ai/docs/observability/scrub_data
code: `litellm/litellm_core_utils/redact_messages.py`, `litellm/litellm_core_utils/litellm_logging.py`
tests: `tests/test_litellm/litellm_core_utils/test_redact_messages.py`
registry: none
verify: set `litellm_settings.turn_off_message_logging: true`, run a call, and confirm the logged payload has redacted messages

### obs.audit_logs: Audit logs (store_audit_logs)
surfaces: config, api | flags: ent, db
docs: https://docs.litellm.ai/docs/proxy/multiple_admins (audit section), https://docs.litellm.ai/docs/data_security
code: `enterprise/litellm_enterprise/proxy/audit_logging_endpoints.py` (GET `/audit`, `/audit/{id}`), `litellm/proxy/hooks/key_management_event_hooks.py`, `litellm/proxy/hooks/user_management_event_hooks.py`
tests: `tests/test_litellm/proxy/hooks/` (audit files)
registry: none
verify: set `general_settings.store_audit_logs: true`, make a /key/update call, then GET /audit and find the change row

## Health and debugging

### obs.health: Liveness / readiness / health endpoints
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/health
code: `litellm/proxy/health_endpoints/_health_endpoints.py` (GET `/health/liveliness`, `/health/readiness`, `/health`, `/health/services`, `/health/history`)
tests: `tests/test_litellm/litellm_core_utils/test_health_check_helpers.py`, `tests/e2e/other/` (lifecycle probes)
registry: other.yaml other.lifecycle.*
verify: curl http://localhost:4000/health/liveliness and /health/readiness; both should return ok while the proxy is up

### obs.debugging: Debug logging, detailed_debug, litellm.log
surfaces: cli, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/debugging, https://docs.litellm.ai/docs/proxy/error_diagnosis, https://docs.litellm.ai/docs/proxy/error_reference
code: `litellm/proxy/proxy_cli.py` (`--detailed_debug` flag), `litellm/_logging.py`, `litellm/proxy/common_utils/debug_utils.py` (`/debug/*` endpoints)
tests: `tests/test_litellm/proxy/` (debug files)
registry: none
verify: boot with `python litellm/proxy/proxy_cli.py --detailed_debug`, run one call, and confirm request-level logs in litellm.log

### obs.profiling: Pyroscope profiling
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/pyroscope_profiling
code: `litellm/proxy/proxy_server.py` (pyroscope init), `litellm/_logging.py`
tests: none found
registry: none
verify: set the pyroscope env vars per the doc, boot, and confirm profiles arrive in your Pyroscope instance

### obs.service_logging: Service callbacks (redis, db latency)
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/observability/callbacks
code: `litellm/integrations/custom_logger.py` (service callback hooks), `litellm/_service_logger.py` if present
tests: `tests/test_litellm/integrations/`
registry: none
verify: register a service callback, run calls that hit redis/db, and confirm latency events are emitted
