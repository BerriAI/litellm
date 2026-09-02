use std::future::Future;

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_core::error::CoreResult;
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = CoreResult<Value>> + Send + 'static> {
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

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
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
    })
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        document: Value,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<Value>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_ocr,
    errors = core_error_to_pyerr,
}
