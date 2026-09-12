use litellm_core::Error;
use std::future::Future;

use litellm_core::ocr::wire::{OcrWireRequest, decode_request};
use pyo3::prelude::*;
use serde_json::Value;

use super::errors::to_pyerr as ocr_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

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
        let request = decode_request(OcrWireRequest {
            model,
            document,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            input_sources,
            timeout_seconds: timeout.map(|value| value.as_secs_f64()),
        })?;
        litellm_core::ocr::ocr(request)
            .await
            .map(|response| response.into_json())
    })
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        document: serde_json::Value,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<serde_json::Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<serde_json::Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        input_sources: Option<serde_json::Value>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
}
