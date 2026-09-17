//! The OCR-specific slice of the legacy `Logging` contract: `update_from_kwargs`,
//! `pre_call`, `post_call` and their payload-free shortcuts. Expires with the contract.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use litellm_python_interop::to_py_preserving_errors as to_py;

use crate::{LegacyCallbacks, PythonLogger};

pub struct OcrLoggingFields {
    pub model: String,
    pub custom_llm_provider: String,
    pub optional_params: Value,
}

pub trait OcrLogger {
    fn update_ocr(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        pre_call: &OcrLoggingFields,
        secret_fields: &[&str],
        url: &str,
    ) -> PyResult<()>;

    fn record_api_call_start(&self, py: Python<'_>) -> PyResult<()>;

    fn pre_ocr(
        &self,
        py: Python<'_>,
        api_key: &Option<Py<PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()>;

    fn post_ocr(
        &self,
        py: Python<'_>,
        original_response: &Value,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()>;
}

impl OcrLogger for PythonLogger {
    fn update_ocr(
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
        for name in custom_pricing_fields(py)? {
            if let Some(value) = kwargs.bind(py).get_item(&name)?
                && !value.is_none()
            {
                params.set_item(name, value)?;
            }
        }
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", &pre_call.custom_llm_provider)?;
        self.object(py)
            .call_method("update_from_kwargs", (), Some(&update))?;
        Ok(())
    }

    fn record_api_call_start(&self, py: Python<'_>) -> PyResult<()> {
        self.object(py).call_method0("record_api_call_start_time")?;
        Ok(())
    }

    fn pre_ocr(
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
            self.record_api_call_start(py)?;
        }
        Ok(())
    }

    fn post_ocr(
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

fn custom_pricing_fields(py: Python<'_>) -> PyResult<Vec<String>> {
    py.import("litellm.types.utils")?
        .getattr("CustomPricingLiteLLMParams")?
        .getattr("model_fields")?
        .cast_into::<PyDict>()?
        .keys()
        .iter()
        .map(|name| name.extract::<String>())
        .collect()
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
