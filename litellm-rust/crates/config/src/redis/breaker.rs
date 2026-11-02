use std::time::Duration;

use litellm_circuit_breaker::Config as BreakerConfig;
use litellm_core_utils::settings::Lookup;
use litellm_python_compat::number;

/// The breaker `RedisCache` builds from the `REDIS_CIRCUIT_BREAKER_*` constants in
/// `litellm/constants.py`, or none when it is disabled. Like `get_env_int` and `get_env_float`,
/// a value that does not parse is warned about and the default is kept.
pub(super) fn breaker(environment: &dyn Lookup) -> Option<BreakerConfig> {
    let enabled = environment
        .get("REDIS_CIRCUIT_BREAKER_ENABLED")
        .is_none_or(|value| value.eq_ignore_ascii_case("true"));
    if !enabled {
        return None;
    }
    let defaults = BreakerConfig::default();
    Some(BreakerConfig {
        failure_threshold: setting(
            environment,
            "REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            |text| number::int(text).and_then(|threshold| u32::try_from(threshold).ok()),
        )
        .unwrap_or(defaults.failure_threshold),
        recovery_timeout: setting(
            environment,
            "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
            |text| {
                number::int(text)
                    .and_then(|seconds| u64::try_from(seconds).ok())
                    .map(Duration::from_secs)
            },
        )
        .unwrap_or(defaults.recovery_timeout),
        timeout_min_duration: setting(
            environment,
            "REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION",
            |text| {
                number::float(text).and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            },
        )
        .unwrap_or(defaults.timeout_min_duration),
    })
}

fn setting<T>(
    environment: &dyn Lookup,
    name: &str,
    parse: impl Fn(&str) -> Option<T>,
) -> Option<T> {
    let raw = environment.get(name).filter(|raw| !raw.trim().is_empty())?;
    let parsed = parse(raw.trim());
    if parsed.is_none() {
        tracing::warn!(
            setting = name,
            "Redis circuit breaker setting is not a valid number, so the default is used like Python"
        );
    }
    parsed
}
