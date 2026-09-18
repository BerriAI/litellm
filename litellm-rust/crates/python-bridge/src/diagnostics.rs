use litellm_host_python::release_count;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub(crate) fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", release_count())?;
    Ok(stats.into_any().unbind())
}

#[cfg(feature = "panic-test")]
#[pyfunction]
pub(crate) fn _panic_for_test() {
    panic!("intentional PyO3 panic smoke test");
}
