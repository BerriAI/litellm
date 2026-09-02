use std::time::Duration;

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::marshal::{optional_object_to_map, optional_timeout};

use super::runtime::{run_async, run_sync};

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
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
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
) -> PyResult<Py<PyAny>> {
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    run_sync(
        py,
        async move {
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
            })
            .await
        },
        core_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
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
) -> PyResult<Bound<'_, PyAny>> {
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    run_async(
        py,
        async move {
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
            })
            .await
        },
        core_error_to_pyerr,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)
}
