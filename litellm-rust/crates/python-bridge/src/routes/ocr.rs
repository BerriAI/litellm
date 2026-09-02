use std::time::Duration;

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_core::error::Error;
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::execution::{run_async, run_sync};
use crate::function_trace::trace_call;
use crate::marshal::{optional_object, optional_object_to_map, optional_timeout};

struct OcrInputs {
    model: String,
    document: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    optional_params: Map<String, Value>,
    timeout: Option<Duration>,
}

#[allow(clippy::too_many_arguments)]
fn marshal_inputs(
    py: Python<'_>,
    model: String,
    document: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<OcrInputs> {
    Ok(OcrInputs {
        model,
        document: from_py(document.bind(py))?,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers: optional_object(py, "extra_headers", extra_headers)?,
        optional_params: optional_object_to_map(py, "optional_params", optional_params)?,
        timeout: optional_timeout(timeout_seconds),
    })
}

async fn call(inputs: OcrInputs) -> Result<Value, Error> {
    run_ocr(OcrRequest {
        model: &inputs.model,
        document: inputs.document,
        api_key: inputs.api_key.as_deref(),
        api_base: inputs.api_base.as_deref(),
        custom_llm_provider: inputs.custom_llm_provider.as_deref(),
        extra_headers: inputs.extra_headers,
        optional_params: inputs.optional_params,
        timeout: inputs.timeout,
        callbacks: Vec::new(),
        guardrails: Vec::new(),
        request_metadata: Default::default(),
        litellm_call_id: None,
    })
    .await
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn ocr(
    py: Python<'_>,
    model: String,
    document: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Py<PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        document,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;
    run_sync(py, trace_call(call(inputs), trace), core_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn aocr(
    py: Python<'_>,
    model: String,
    document: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        document,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;
    run_async(py, trace_call(call(inputs), trace), core_error_to_pyerr)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)
}
