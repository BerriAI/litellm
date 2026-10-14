use litellm_host_python::{release_count, runtime_started};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

#[pyfunction]
pub(crate) fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", release_count())?;
    Ok(stats.into_any().unbind())
}

/// True once this process has started the native runtime, which does not survive `fork()`.
#[pyfunction]
pub(crate) fn process_state_started() -> bool {
    runtime_started()
}

/// Declares that this process only forks workers: from now on every native route raises here,
/// so the runtime can never start. Raises if it already has. Forked workers are unaffected.
#[pyfunction]
pub(crate) fn reserve_process_for_forking() -> PyResult<()> {
    litellm_host_python::reserve_process_for_forking()
        .map_err(|_| PyRuntimeError::new_err("the native runtime already started in this process"))
}

#[cfg(feature = "panic-test")]
#[pyfunction]
pub(crate) fn _panic_for_test() {
    panic!("intentional PyO3 panic smoke test");
}
