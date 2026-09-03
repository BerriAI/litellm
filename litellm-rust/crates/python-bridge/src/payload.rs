use litellm_core::http_utils::body::{JsonPayload, SharedText};
use litellm_python_interop::{bytes_from_py, from_py, text_bytes_from_py};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple};

pub(crate) fn payload_from_py(value: &Bound<'_, PyAny>) -> PyResult<JsonPayload> {
    extract(value, 0, false)
}

pub(crate) fn audio_payload_from_py(value: &Bound<'_, PyAny>) -> PyResult<JsonPayload> {
    extract(value, 0, true)
}

fn extract(value: &Bound<'_, PyAny>, depth: usize, audio: bool) -> PyResult<JsonPayload> {
    if depth > litellm_core::constants::JSON_PAYLOAD_MAX_DEPTH {
        return Err(PyValueError::new_err(
            "request nesting exceeds the JSON depth limit",
        ));
    }
    if value.is_instance_of::<PyString>() {
        return SharedText::new(text_bytes_from_py(value)?)
            .map(JsonPayload::String)
            .map_err(|_| PyValueError::new_err("invalid UTF-8 string"));
    }
    if audio
        && depth == 1
        && (value.is_instance_of::<PyBytes>() || value.is_instance_of::<PyByteArray>())
    {
        return Ok(JsonPayload::Base64(bytes_from_py(value)?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return dict
            .iter()
            .map(|(key, value)| {
                let key = key.extract::<String>()?;
                let value = extract(&value, depth + 1, audio && depth == 0 && key == "data")?;
                Ok((key, value))
            })
            .collect::<PyResult<_>>()
            .map(JsonPayload::Object);
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        return value
            .try_iter()?
            .map(|item| extract(&item?, depth + 1, false))
            .collect::<PyResult<_>>()
            .map(JsonPayload::Array);
    }
    from_py::<serde_json::Value>(value).map(JsonPayload::from)
}
