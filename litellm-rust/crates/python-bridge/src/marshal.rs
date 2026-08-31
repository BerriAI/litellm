use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub fn from_py<T>(value: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: DeserializeOwned,
{
    pythonize::depythonize(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn to_py<T>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>>
where
    T: Serialize + ?Sized,
{
    pythonize::pythonize(py, value)
        .map(Bound::unbind)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn py_to_json(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    from_py(value)
}
