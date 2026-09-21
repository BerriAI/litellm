use litellm_python_interop::release_count;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", release_count())?;
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
