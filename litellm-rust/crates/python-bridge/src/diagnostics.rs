use litellm_python_interop::release_count;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", release_count())?;
    Ok(stats.into_any().unbind())
}

#[pyfunction]
fn _debug_setup(
    py: Python<'_>,
    call_type: String,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    start: Py<PyAny>,
    asynchronous: bool,
) -> PyResult<(Py<PyAny>, Py<PyDict>)> {
    let leaked: &'static str = Box::leak(call_type.into_boxed_str());
    let result = crate::lifecycle::debug_setup(
        py,
        leaked,
        &args.unbind(),
        &kwargs.unbind(),
        &start,
        asynchronous,
    )?;
    Ok(result)
}

#[cfg(feature = "panic-test")]
#[pyfunction]
fn _panic_for_test() {
    panic!("intentional PyO3 panic smoke test");
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_debug_setup, module)?)?;
    #[cfg(feature = "panic-test")]
    module.add_function(wrap_pyfunction!(_panic_for_test, module)?)?;
    Ok(())
}
