use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_core::error::CoreResult;
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, object_or_empty};
use crate::routes::BridgeRoute;

struct OcrCall {
    options: RouteOptions,
    document: Value,
    optional_params: Map<String, Value>,
}

impl BridgeRoute<OcrInputs> for OcrCall {
    type Output = Value;

    fn from_python(py: Python<'_>, inputs: OcrInputs) -> PyResult<Self> {
        Ok(Self {
            options: RouteOptions::from_python(
                py,
                inputs.model,
                inputs.api_key,
                inputs.api_base,
                inputs.custom_llm_provider,
                inputs.extra_headers,
                inputs.timeout_seconds,
            )?,
            document: from_py(inputs.document.bind(py))?,
            optional_params: object_or_empty(py, "optional_params", inputs.optional_params)?,
        })
    }

    async fn run(self) -> CoreResult<Value> {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = self.options;
        run_ocr(OcrRequest {
            model: &model,
            document: self.document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params: self.optional_params,
            timeout,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        })
        .await
    }
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        model: String,
        document: Py<PyAny>,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        optional_params: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    },
    call = OcrCall,
    errors = core_error_to_pyerr,
}
