use litellm_core::Error;
use litellm_core::ocr::{
    OcrClient,
    wire::{OcrWireRequest, decode_request},
};
use serde_json::Value;

mod types;

pub use types::OcrRequest;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(request: OcrRequest<'_>) -> Result<Value, Error> {
    core_ocr(request).await
}

async fn core_ocr(request: OcrRequest<'_>) -> Result<Value, Error> {
    validate_host_hooks(&request)?;
    let client = OcrClient::new(crate::client::http_client().clone())?;
    let core_request = decode_request(OcrWireRequest {
        model: request.model.to_string(),
        document: request.document,
        api_key: request.api_key.map(str::to_string),
        api_base: request.api_base.map(str::to_string),
        custom_llm_provider: request.custom_llm_provider.map(str::to_string),
        extra_headers: request.extra_headers,
        optional_params: request.optional_params,
        input_sources: Default::default(),
        timeout_seconds: request.timeout.map(|timeout| timeout.as_secs_f64()),
    })?;
    client
        .perform(core_request)
        .await
        .map(|response| response.into_json())
}

fn validate_host_hooks(request: &OcrRequest<'_>) -> Result<(), Error> {
    if !request.guardrails.is_empty() {
        return Err(Error::Unsupported(
            "OCR host guardrails are not wired to the core path",
        ));
    }
    if !request.callbacks.is_empty() {
        return Err(Error::Unsupported(
            "OCR host callbacks are not wired to the core path",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_core::ocr::wire::is_supported_request;
    use serde_json::{Map, json};

    use super::{OcrRequest, validate_host_hooks};
    use crate::integrations::custom_guardrail::{CustomGuardrail, GuardrailEventHook};
    use crate::integrations::custom_logger::CustomLogger;

    struct TestGuardrail;

    impl CustomGuardrail for TestGuardrail {
        fn guardrail_name(&self) -> &str {
            "test"
        }

        fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
            &[]
        }
    }

    struct TestLogger;

    impl CustomLogger for TestLogger {}

    fn request() -> OcrRequest<'static> {
        OcrRequest {
            model: "model",
            document: json!({"type":"image_url","image_url":"data:image/png;base64,YQ=="}),
            api_key: None,
            api_base: None,
            custom_llm_provider: Some("mistral"),
            extra_headers: None,
            optional_params: Map::new(),
            timeout: None,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        }
    }

    #[test]
    fn core_activation_includes_migrated_providers() {
        assert!(is_supported_request("model", Some("mistral")));
        assert!(is_supported_request("pixtral-12b", Some("azure_ai")));
        assert!(is_supported_request(
            "doc-intelligence/prebuilt-layout",
            Some("azure_ai")
        ));
        assert!(is_supported_request("parse-v3", Some("reducto")));
        assert!(is_supported_request("mistral-ocr", Some("vertex_ai")));
        assert!(is_supported_request("deepseek-ocr", Some("vertex_ai")));
    }

    #[test]
    fn core_path_rejects_unwired_guardrails() {
        let request = OcrRequest {
            guardrails: vec![Arc::new(TestGuardrail)],
            ..request()
        };
        let error = validate_host_hooks(&request).unwrap_err();
        assert!(error.to_string().contains("guardrails are not wired"));
    }

    #[test]
    fn core_path_rejects_unwired_callbacks() {
        let request = OcrRequest {
            callbacks: vec![Arc::new(TestLogger)],
            ..request()
        };
        let error = validate_host_hooks(&request).unwrap_err();
        assert!(error.to_string().contains("callbacks are not wired"));
    }
}
