use std::time::Duration;

use litellm_redis_store::RouterTiers;
use pyo3::prelude::*;

use crate::{
    coercion::{Field, FieldSpec, ProjectionError},
    python_settings::{PythonSettings, Snapshot},
};

const COOLDOWN_REFRESH: FieldSpec<Duration> =
    FieldSpec::new("cooldown_redis_read_interval", seconds);
const USAGE_REFRESH: FieldSpec<Duration> = FieldSpec::new("usage_redis_read_interval", seconds);
const TRACKED_READS: FieldSpec<usize> = FieldSpec::new("max_redis_batch_cache_size", count);
const MEMORY_TTL: FieldSpec<Option<Duration>> =
    FieldSpec::new("default_in_memory_ttl", optional_seconds);

/// The tier values this Python process runs with, from its environment and `litellm` globals.
pub(crate) fn router_tier_policies(py: Python<'_>) -> PyResult<RouterTiers> {
    Ok(project(&PythonSettings::RouterTiers.read(py)?)?)
}

fn project(snapshot: &Snapshot<'_>) -> Result<RouterTiers, ProjectionError> {
    Ok(RouterTiers {
        cooldown_refresh: snapshot.read(&COOLDOWN_REFRESH)?,
        usage_refresh: snapshot.read(&USAGE_REFRESH)?,
        tracked_reads: snapshot.read(&TRACKED_READS)?,
        default_in_memory_ttl: snapshot.read(&MEMORY_TTL)?,
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

fn optional_seconds(field: &Field<'_>) -> Result<Option<Duration>, ProjectionError> {
    match field.value().is_none() {
        true => Ok(None),
        false => seconds(field).map(Some),
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
    use rstest::rstest;

    use super::project;
    use crate::python_settings::PythonSettings;

    fn projected(py: Python<'_>, fields: &str) -> Result<litellm_redis_store::RouterTiers, String> {
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

    #[rstest]
    fn the_host_values_are_each_familys_refresh_and_tracked_reads() {
        Python::initialize();
        Python::attach(|py| {
            let policies = projected(
                py,
                "cooldown_redis_read_interval=2.5, usage_redis_read_interval=30, max_redis_batch_cache_size=42, default_in_memory_ttl=None",
            )
            .unwrap();
            assert_eq!(
                policies,
                litellm_redis_store::RouterTiers {
                    cooldown_refresh: Duration::from_millis(2500),
                    usage_refresh: Duration::from_secs(30),
                    tracked_reads: 42,
                    default_in_memory_ttl: None,
                }
            );
        });
    }

    #[rstest]
    fn a_set_default_in_memory_ttl_reaches_every_familys_backfill() {
        Python::initialize();
        Python::attach(|py| {
            let policies = projected(
                py,
                "cooldown_redis_read_interval=1, usage_redis_read_interval=10, max_redis_batch_cache_size=1000, default_in_memory_ttl=7.5",
            )
            .unwrap();
            assert_eq!(
                (
                    policies.cooldowns().default_in_memory_ttl,
                    policies.usage().default_in_memory_ttl,
                    policies.default_in_memory_ttl,
                ),
                (
                    Some(Duration::from_millis(7500)),
                    Some(Duration::from_millis(7500)),
                    Some(Duration::from_millis(7500)),
                )
            );
        });
    }

    #[rstest]
    fn a_value_python_would_not_run_with_is_a_configuration_error() {
        Python::initialize();
        Python::attach(|py| {
            let negative = projected(
                py,
                "cooldown_redis_read_interval=-1, usage_redis_read_interval=10, max_redis_batch_cache_size=1000, default_in_memory_ttl=None",
            )
            .unwrap_err();
            let text = projected(
                py,
                "cooldown_redis_read_interval=1, usage_redis_read_interval=10, max_redis_batch_cache_size='many', default_in_memory_ttl=None",
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
