use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use litellm_core::ocr::LiteLLMOcrResponse;
use litellm_core::ocr::hooks::OcrPreCallRequest;
use litellm_python_interop::to_py_preserving_errors as to_py;

use crate::lifecycle::PythonLogger;

pub(super) struct OcrLoggingFields {
    model: String,
    custom_llm_provider: String,
    optional_params: Value,
}

impl From<&OcrPreCallRequest> for OcrLoggingFields {
    fn from(request: &OcrPreCallRequest) -> Self {
        Self {
            model: request.model.clone(),
            custom_llm_provider: request.custom_llm_provider.clone(),
            optional_params: request.optional_params.clone(),
        }
    }
}

impl PythonLogger {
    pub(super) fn update_ocr(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        pre_call: &OcrLoggingFields,
        url: &str,
    ) -> PyResult<()> {
        let redact = py
            .import("litellm.rust_bridge.ocr")?
            .getattr("redact_logging_params")?;
        let update = PyDict::new(py);
        update.set_item("kwargs", redact.call1((kwargs,))?.cast_into::<PyDict>()?)?;
        update.set_item("model", &pre_call.model)?;
        update.set_item(
            "optional_params",
            redact
                .call1((to_py(py, &pre_call.optional_params)?,))?
                .cast_into::<PyDict>()?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", url)?;
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", &pre_call.custom_llm_provider)?;
        self.object(py)
            .call_method("update_from_kwargs", (), Some(&update))?;
        Ok(())
    }

    pub(crate) fn pre_ocr(
        &self,
        py: Python<'_>,
        api_key: &Option<Py<PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        additional.set_item("api_base", url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", api_key)?;
        kwargs.set_item("additional_args", additional)?;
        self.object(py).call_method("pre_call", (), Some(&kwargs))?;
        Ok(())
    }

    pub(crate) fn post_ocr(
        &self,
        py: Python<'_>,
        original_response: &Value,
        body: &Option<Py<PyDict>>,
        headers: &Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("original_response", to_py(py, original_response)?)?;
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        kwargs.set_item("additional_args", additional)?;
        self.object(py)
            .call_method("post_call", (), Some(&kwargs))?;
        Ok(())
    }
}

pub(super) fn response(py: Python<'_>, response: &LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("_response")?
        .call1((to_py(py, response)?,))
        .map(Bound::unbind)
}

pub(super) fn map_failure(
    py: Python<'_>,
    error: &Py<PyBaseException>,
    request: &Bound<'_, PyAny>,
    provider: &str,
) -> PyResult<Py<PyBaseException>> {
    Ok(py
        .import("litellm.rust_bridge.ocr_lifecycle")?
        .getattr("map_failure")?
        .call1((error, request, provider))?
        .extract()?)
}
