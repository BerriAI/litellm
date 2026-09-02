use std::time::Duration;

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_python_interop::{from_py, release_gil, to_py};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::function_trace::trace_call;
use crate::marshal::{optional_object_to_map, optional_timeout};

type MarshaledOcrInputs = (
    Value,
    Option<Map<String, Value>>,
    Map<String, Value>,
    Option<Duration>,
);

fn marshal_inputs(
    py: Python<'_>,
    document: Py<PyAny>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<MarshaledOcrInputs> {
    let document = from_py(document.bind(py))?;
    let extra_headers = match extra_headers {
        Some(headers) => Some(optional_object_to_map(py, "extra_headers", Some(headers))?),
        None => None,
    };
    let optional_params = optional_object_to_map(py, "optional_params", optional_params)?;
    let timeout = optional_timeout(timeout_seconds);

    Ok((document, extra_headers, optional_params, timeout))
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
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    let result = release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(trace_call(
            run_ocr(OcrRequest {
                model: &model,
                document,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                optional_params,
                timeout,
                callbacks: Vec::new(),
                guardrails: Vec::new(),
                request_metadata: Default::default(),
                litellm_call_id: None,
            }),
            trace,
        ))
    });

    match result {
        Ok(value) => to_py(py, &value),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
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
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let value = trace_call(
            run_ocr(OcrRequest {
                model: &model,
                document,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                optional_params,
                timeout,
                callbacks: Vec::new(),
                guardrails: Vec::new(),
                request_metadata: Default::default(),
                litellm_call_id: None,
            }),
            trace,
        )
        .await
        .map_err(core_error_to_pyerr)?;

        Python::attach(|py| to_py(py, &value))
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)
}
