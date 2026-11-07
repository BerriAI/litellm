# Health check driven routing

## Startup behavior

Until the first background cycle completes, every deployment is treated as healthy. A request in that window can still be sent to a deployment that is already down, and it waits for that deployment's request timeout before falling back.

The cycle writes those results only after its probes finish. Each probe waits up to `model_info.health_check_timeout` seconds, or `HEALTH_CHECK_TIMEOUT_SECONDS` when that field is unset (60 seconds by default, overridable with the `HEALTH_CHECK_TIMEOUT_SECONDS` environment variable). A lower `model_info.health_check_timeout` shortens this window.

When `allowed_fails_policy` is set and `background_health_check_model_groups` is not, cooldown is what removes a deployment. That counter is also empty until the first cycle records a failure, so the same window applies.
