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
        secret_fields: &[&str],
        url: &str,
    ) -> PyResult<()> {
        let update = PyDict::new(py);
        update.set_item("kwargs", redact(py, kwargs.bind(py), secret_fields)?)?;
        update.set_item("model", &pre_call.model)?;
        update.set_item(
            "optional_params",
            redact(
                py,
                &to_py(py, &pre_call.optional_params)?
                    .into_bound(py)
                    .cast_into::<PyDict>()?,
                secret_fields,
            )?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", url)?;
        for name in ["logger_fn", "litellm_request_debug"] {
            if let Some(value) = kwargs.bind(py).get_item(name)? {
                params.set_item(name, value)?;
            }
        }
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
        kwargs.set_item("additional_args", &additional)?;
        if self.callbacks_needed(py, "input")? {
            self.object(py).call_method("pre_call", (), Some(&kwargs))?;
        } else {
            self.object(py)
                .call_method("_pre_call", (), Some(&kwargs))?;
            self.object(py).call_method0("record_api_call_start_time")?;
        }
        Ok(())
    }

    pub(crate) fn post_ocr(
        &self,
        py: Python<'_>,
        original_response: &Value,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        if self.callbacks_needed(py, "input")? {
            let kwargs = PyDict::new(py);
            kwargs.set_item("original_response", to_py(py, original_response)?)?;
            kwargs.set_item("additional_args", &additional)?;
            self.object(py)
                .call_method("post_call", (), Some(&kwargs))?;
        } else {
            let response = py
                .import("json")?
                .call_method1("dumps", (to_py(py, original_response)?,))?;
            self.object(py).call_method1(
                "record_post_call",
                (response, py.None(), py.None(), additional),
            )?;
        }
        Ok(())
    }
}

fn redact(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    secret_fields: &[&str],
) -> PyResult<Py<PyDict>> {
    let redacted = PyDict::new(py);
    for (name, value) in params {
        let name = name.extract::<String>()?;
        if name == "proxy_server_request" {
            continue;
        }
        if secret_fields.contains(&name.as_str()) {
            redacted.set_item(name, "****")?;
        } else {
            redacted.set_item(name, value)?;
        }
    }
    Ok(redacted.unbind())
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
