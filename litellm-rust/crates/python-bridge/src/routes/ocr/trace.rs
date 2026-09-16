use litellm_core::ocr::Error;
use std::future::Future;

use litellm_core::ocr::LiteLLMOcrRequest;
use pyo3::prelude::*;
use serde_json::Value;

use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::request::BridgeOcrRequest;
use crate::execution::{run_async, run_sync};
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

struct OcrInputs {
    model: String,
    document: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Value>,
    optional_params: Option<Value>,
    input_sources: Option<Value>,
    timeout_seconds: Option<f64>,
}

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let document = inputs.document;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;
    let input_sources = inputs
        .input_sources
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .unwrap_or_default();

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        let request = LiteLLMOcrRequest::try_from(BridgeOcrRequest {
            model,
            document,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params: optional_params.into(),
            input_sources,
            timeout_seconds: timeout.map(|value| value.as_secs_f64()),
        })?;
        litellm_core::ocr::ocr(request)
            .await
            .map(|response| response.into_json())
    })
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, input_sources=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn ocr(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] document: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] input_sources: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    run_sync(
        py,
        crate::function_trace::capture(prepare_ocr(OcrInputs {
            model,
            document,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            input_sources,
            timeout_seconds,
        })?),
        ocr_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, input_sources=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn aocr(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] document: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] input_sources: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    run_async(
        py,
        crate::function_trace::capture(prepare_ocr(OcrInputs {
            model,
            document,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            input_sources,
            timeout_seconds,
        })?),
        ocr_error_to_pyerr,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::super::add_function(module, wrap_pyfunction!(ocr, module)?)?;
    super::super::add_function(module, wrap_pyfunction!(aocr, module)?)
}
