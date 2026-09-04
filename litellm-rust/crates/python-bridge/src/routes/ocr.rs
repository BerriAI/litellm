use litellm_core::Error;
use std::future::Future;

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_core::routing_utils::provider::get_custom_llm_provider;
use pyo3::prelude::*;
use serde_json::Value;

use crate::constants::RUST_OCR_PROVIDERS;
use crate::errors::ocr_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

#[pyfunction]
#[pyo3(signature = (model, optional_params=None, custom_llm_provider=None))]
fn ocr_decline(
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    custom_llm_provider: Option<String>,
) -> PyResult<Option<String>> {
    let optional_params = object_or_empty("optional_params", optional_params)?;
    let provider = get_custom_llm_provider(&model, custom_llm_provider.as_deref())
        .map(|info| info.custom_llm_provider)
        .unwrap_or("mistral");
    if !RUST_OCR_PROVIDERS.contains(&provider) {
        return Ok(Some(format!("provider {provider} is not supported")));
    }
    if optional_params.get("req_format").and_then(Value::as_str) == Some("native") {
        return Ok(Some("req_format=native is not supported".to_string()));
    }
    Ok(None)
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
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
    extra = [ocr_decline],
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decline_gate_owns_provider_and_request_capability() {
        assert_eq!(
            ocr_decline("reducto/parse-v3".to_string(), None, None).expect("gate should evaluate"),
            Some("provider reducto is not supported".to_string())
        );
        assert_eq!(
            ocr_decline(
                "mistral-ocr-latest".to_string(),
                Some(json!({"req_format": "native"})),
                Some("mistral".to_string()),
            )
            .expect("gate should evaluate"),
            Some("req_format=native is not supported".to_string())
        );
        assert_eq!(
            ocr_decline(
                "mistral/mistral-ocr-latest".to_string(),
                Some(json!({})),
                None,
            )
            .expect("gate should evaluate"),
            None
        );
    }
}
