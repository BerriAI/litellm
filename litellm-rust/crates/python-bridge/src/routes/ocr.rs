use std::future::Future;

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_core::Error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::errors::ocr_error_to_pyerr;
use crate::marshal::optional_timeout;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BridgeOcrRequest {
    model: String,
    document: Value,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    custom_llm_provider: Option<String>,
    #[serde(default)]
    extra_headers: Option<Map<String, Value>>,
    #[serde(default)]
    optional_params: Map<String, Value>,
    #[serde(default)]
    timeout_seconds: Option<f64>,
}

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let request: BridgeOcrRequest = serde_json::from_value(inputs.request)
        .map_err(|error| PyValueError::new_err(format!("invalid OCR request: {error}")))?;

    Ok(async move {
        run_ocr(OcrRequest {
            model: &request.model,
            document: request.document,
            api_key: request.api_key.as_deref(),
            api_base: request.api_base.as_deref(),
            custom_llm_provider: request.custom_llm_provider.as_deref(),
            extra_headers: request.extra_headers,
            optional_params: request.optional_params,
            timeout: optional_timeout(request.timeout_seconds),
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        })
        .await
    })
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        request: serde_json::Value,
    },
    optional = {},
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
}
