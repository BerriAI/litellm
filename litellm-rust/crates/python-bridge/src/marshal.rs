use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

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

pub fn from_json_str<T>(json: &str) -> Result<T, String>
where
    T: DeserializeOwned,
{
    serde_json::from_str(json).map_err(|error| error.to_string())
}

pub fn to_json_string<T>(value: &T) -> Result<String, String>
where
    T: Serialize + ?Sized,
{
    serde_json::to_string(value).map_err(|error| error.to_string())
}
