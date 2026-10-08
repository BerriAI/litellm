use litellm_host_python::to_py;
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn control_params(py: Python<'_>) -> PyResult<Py<PyAny>> {
    to_py(py, litellm_core_utils::params::control_params())
}
