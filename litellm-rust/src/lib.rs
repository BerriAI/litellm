//! Crate root.
//!
//! Exposes the pure transform/pipeline modules, and — when built with the
//! `python` feature — a PyO3 extension module `litellm_rust` with a single
//! sync `ocr(...)` function.

pub mod error;
pub mod llms;
pub mod pipeline;

#[cfg(feature = "python")]
mod python {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyString};
    use serde_json::{Map, Value};

    use crate::error::OcrError;
    use crate::llms::base_llm::ocr::transformation::OcrRequest;
    use crate::pipeline::ocr_blocking;

    /// Convert a Python object into a `serde_json::Value` by round-tripping
    /// through the `json` module's `dumps`. Keeps conversion logic simple and
    /// faithful to Python's own JSON encoding.
    fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
        let json = py.import("json")?;
        let dumped = json.call_method1("dumps", (obj,))?;
        let s: String = dumped.extract()?;
        serde_json::from_str(&s)
            .map_err(|e| PyValueError::new_err(format!("Failed to encode argument as JSON: {e}")))
    }

    /// Convert a `serde_json::Value` into a Python object via `json.loads`.
    fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
        let s = serde_json::to_string(value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize response: {e}")))?;
        let json = py.import("json")?;
        let obj = json.call_method1("loads", (PyString::new(py, &s),))?;
        Ok(obj.unbind())
    }

    /// Map an `OcrError` to the appropriate Python exception type.
    fn ocr_error_to_pyerr(err: OcrError) -> PyErr {
        match err {
            // Missing API key etc. -> ValueError (matches the Python code).
            OcrError::Auth(msg) => PyValueError::new_err(msg),
            // Bad document type etc. -> ValueError.
            OcrError::Transform(msg) => PyValueError::new_err(msg),
            // HTTP/network/parse failures -> RuntimeError carrying details.
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }

    /// Perform a Mistral OCR call.
    ///
    /// Mirrors:
    /// `litellm_rust.ocr(model, document, api_key, api_base, optional_params) -> dict`
    #[pyfunction]
    #[pyo3(signature = (model, document, api_key, api_base, optional_params))]
    pub fn ocr(
        py: Python<'_>,
        model: String,
        document: &Bound<'_, PyAny>,
        api_key: Option<String>,
        api_base: Option<String>,
        optional_params: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let document_value = py_to_value(py, document)?;

        let params_value = py_to_value(py, optional_params.as_any())?;
        let optional_params: Map<String, Value> = match params_value {
            Value::Object(map) => map,
            _ => return Err(PyValueError::new_err("optional_params must be a dict")),
        };

        let req = OcrRequest {
            model,
            document: document_value,
            api_key,
            api_base,
            optional_params,
        };

        // Release the GIL during the blocking HTTP call.
        let result = py.allow_threads(|| ocr_blocking(req));

        match result {
            Ok(response) => {
                let value = serde_json::to_value(&response).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to serialize response: {e}"))
                })?;
                value_to_py(py, &value)
            }
            Err(err) => Err(ocr_error_to_pyerr(err)),
        }
    }

    #[pymodule]
    fn litellm_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(ocr, m)?)?;
        Ok(())
    }
}
