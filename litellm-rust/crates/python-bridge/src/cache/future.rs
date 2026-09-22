use litellm_host_python::to_py;
use pyo3::prelude::*;

pub(super) fn ready_none(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    ready_value(py, &())
}

pub(super) fn ready_value<'py, T: serde::Serialize>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    let future = py
        .import("asyncio")?
        .call_method0("get_running_loop")?
        .call_method0("create_future")?;
    future.call_method1("set_result", (to_py(py, value)?,))?;
    Ok(future)
}
