use litellm_python_interop::{acquisition_count, release_count};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", release_count())?;
    stats.set_item("acquisitions", acquisition_count())?;
    Ok(stats.into_any().unbind())
}

#[cfg(feature = "panic-test")]
#[pyfunction]
fn _panic_for_test() {
    panic!("intentional PyO3 panic smoke test");
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    #[cfg(feature = "panic-test")]
    module.add_function(wrap_pyfunction!(_panic_for_test, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn stat(py: Python<'_>, key: &str) -> u64 {
        gil_stats(py)
            .expect("stats should build")
            .into_bound(py)
            .cast_into::<PyDict>()
            .expect("stats should be a dict")
            .get_item(key)
            .expect("key should resolve")
            .expect("key should be present")
            .extract()
            .expect("stat should be an integer")
    }

    #[test]
    fn gil_stats_reports_both_interop_counters() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(stat(py, "releases"), release_count());
            assert_eq!(stat(py, "acquisitions"), acquisition_count());
        });
    }

    #[test]
    fn gil_stats_counts_a_sync_route_run_on_both_counters() {
        Python::initialize();
        Python::attach(|py| {
            let releases_before = stat(py, "releases");
            let acquisitions_before = stat(py, "acquisitions");

            crate::execution::run_sync(
                py,
                async {
                    tokio::time::sleep(Duration::from_millis(120)).await;
                    Ok(true)
                },
                crate::errors::core_error_to_pyerr,
            )
            .expect("sync route should complete");

            assert!(stat(py, "releases") > releases_before);
            assert!(stat(py, "acquisitions") > acquisitions_before);
        });
    }
}
