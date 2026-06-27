use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

pub(crate) fn py_to_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let encoded: String = json.call_method1("dumps", (value,))?.extract()?;
    serde_json::from_str(&encoded).map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn json_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    let json = py.import("json")?;
    let encoded =
        serde_json::to_string(&value).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(json.call_method1("loads", (encoded,))?.unbind())
}

pub(crate) fn py_attr_string(py: Python<'_>, obj: &Py<PyAny>, attr_name: &str) -> Option<String> {
    let attr = obj.bind(py).getattr(attr_name).ok()?;
    if attr.is_none() {
        return None;
    }
    if let Ok(value_attr) = attr.getattr("value") {
        if let Ok(value) = value_attr.extract::<String>() {
            return Some(value);
        }
    }
    attr.extract::<String>()
        .ok()
        .or_else(|| attr.str().ok().map(|value| value.to_string()))
}

pub(crate) fn py_datetime_from_epoch_seconds(
    py: Python<'_>,
    timestamp: f64,
) -> PyResult<Py<PyAny>> {
    let datetime = py.import("datetime")?.getattr("datetime")?;
    Ok(datetime
        .call_method1("fromtimestamp", (timestamp,))?
        .unbind())
}

pub(crate) async fn call_python_awaitable(
    obj: Py<PyAny>,
    method_name: &'static str,
    args: Vec<Py<PyAny>>,
    kwargs: Option<Py<PyDict>>,
) -> PyResult<Py<PyAny>> {
    let awaitable = Python::with_gil(|py| {
        let callable = obj.bind(py).getattr(method_name)?;
        let args_tuple = pyo3::types::PyTuple::new(py, args.iter().map(|arg| arg.bind(py)))?;
        callable
            .call(args_tuple, kwargs.as_ref().map(|dict| dict.bind(py)))
            .map(|value| value.unbind())
    })?;
    let future =
        Python::with_gil(|py| pyo3_async_runtimes::tokio::into_future(awaitable.into_bound(py)))?;
    future.await
}
