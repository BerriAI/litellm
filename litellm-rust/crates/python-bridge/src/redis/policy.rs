use std::time::Duration;

use litellm_cache::TierPolicy;
use litellm_redis_store::{router_cooldowns, router_usage};
use pyo3::prelude::*;

use crate::{
    coercion::{Field, FieldSpec, ProjectionError},
    python_settings::{PythonSettings, Snapshot},
};

const COOLDOWN_REFRESH: FieldSpec<Duration> =
    FieldSpec::new("cooldown_redis_read_interval", seconds);
const USAGE_REFRESH: FieldSpec<Duration> = FieldSpec::new("usage_redis_read_interval", seconds);
const TRACKED_READS: FieldSpec<usize> = FieldSpec::new("max_redis_batch_cache_size", count);

/// The tier values this Python process runs with. `POLICY` in each family holds `main`'s
/// defaults, the host reports what its environment and `litellm` globals changed.
#[derive(Debug)]
pub(crate) struct RouterTierPolicies {
    pub(crate) cooldowns: TierPolicy,
    pub(crate) usage: TierPolicy,
}

pub(crate) fn router_tier_policies(py: Python<'_>) -> PyResult<RouterTierPolicies> {
    Ok(project(&PythonSettings::RouterTiers.read(py)?)?)
}

fn project(snapshot: &Snapshot<'_>) -> Result<RouterTierPolicies, ProjectionError> {
    let tracked_reads = snapshot.read(&TRACKED_READS)?;
    Ok(RouterTierPolicies {
        cooldowns: TierPolicy {
            refresh_after: snapshot.read(&COOLDOWN_REFRESH)?,
            tracked_reads,
            ..router_cooldowns::POLICY
        },
        usage: TierPolicy {
            refresh_after: snapshot.read(&USAGE_REFRESH)?,
            tracked_reads,
            ..router_usage::POLICY
        },
    })
}

fn seconds(field: &Field<'_>) -> Result<Duration, ProjectionError> {
    match field.value().extract::<f64>() {
        Ok(seconds) if seconds.is_finite() && seconds >= 0.0 => {
            Ok(Duration::from_secs_f64(seconds))
        }
        _ => Err(field.invalid("a non-negative number of seconds")),
    }
}

fn count(field: &Field<'_>) -> Result<usize, ProjectionError> {
    field
        .value()
        .extract::<usize>()
        .map_err(|_| field.invalid("a non-negative integer"))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pyo3::{prelude::*, types::PyDict};

    use super::project;
    use crate::python_settings::PythonSettings;

    fn projected(py: Python<'_>, fields: &str) -> Result<super::RouterTierPolicies, String> {
        let locals = PyDict::new(py);
        let source = std::ffi::CString::new(format!(
            "import types\nsettings = types.SimpleNamespace({fields})"
        ))
        .unwrap();
        py.run(&source, Some(&locals), Some(&locals)).unwrap();
        let snapshot =
            PythonSettings::RouterTiers.snapshot(locals.get_item("settings").unwrap().unwrap());
        project(&snapshot).map_err(|error| PyErr::from(error).to_string())
    }

    #[test]
    fn the_host_values_replace_only_the_refresh_and_tracked_reads_of_mains_defaults() {
        Python::initialize();
        Python::attach(|py| {
            let policies = projected(
                py,
                "cooldown_redis_read_interval=2.5, usage_redis_read_interval=30, max_redis_batch_cache_size=42",
            )
            .unwrap();
            assert_eq!(
                policies.cooldowns.refresh_after,
                Duration::from_millis(2500)
            );
            assert_eq!(policies.usage.refresh_after, Duration::from_secs(30));
            assert_eq!(policies.cooldowns.tracked_reads, 42);
            assert_eq!(policies.usage.tracked_reads, 42);
            assert_eq!(
                policies.cooldowns.backfill_ttl,
                litellm_redis_store::router_cooldowns::POLICY.backfill_ttl
            );
            assert_eq!(
                policies.usage.backfill_ttl,
                litellm_redis_store::router_usage::POLICY.backfill_ttl
            );
        });
    }

    #[test]
    fn a_value_python_would_not_run_with_is_a_configuration_error() {
        Python::initialize();
        Python::attach(|py| {
            let negative = projected(
                py,
                "cooldown_redis_read_interval=-1, usage_redis_read_interval=10, max_redis_batch_cache_size=1000",
            )
            .unwrap_err();
            let text = projected(
                py,
                "cooldown_redis_read_interval=1, usage_redis_read_interval=10, max_redis_batch_cache_size='many'",
            )
            .unwrap_err();
            assert!(
                negative.contains("cooldown_redis_read_interval"),
                "{negative}"
            );
            assert!(text.contains("max_redis_batch_cache_size"), "{text}");
        });
    }
}
