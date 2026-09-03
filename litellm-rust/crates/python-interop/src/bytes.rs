use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

pub fn bytes_from_py(value: &Bound<'_, PyAny>) -> PyResult<Bytes> {
    Ok(value.extract()?)
}

pub fn text_bytes_from_py(value: &Bound<'_, PyAny>) -> PyResult<Bytes> {
    Ok(Bytes::from_owner(value.extract::<PyBackedStr>()?))
}
