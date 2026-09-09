use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::ocr::prepare::prepare_ocr_provider;

use super::hooks::OcrLifecycleHooks;
use super::types::{OcrRequest, PreparedOcrRequest};
use crate::integrations::custom_guardrail::CustomGuardrailRunner;
use crate::integrations::custom_logger::CustomLoggerRunner;

pub(crate) struct PreparedOcrCall {
    pub(crate) request: PreparedOcrRequest,
    pub(crate) hooks: OcrLifecycleHooks,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn prepare_ocr_call(request: OcrRequest<'_>) -> PreparedOcrCall {
    let call_id = request
        .litellm_call_id
        .map(str::to_string)
        .unwrap_or_else(new_ocr_call_id);
    let prepared_provider = prepare_ocr_provider(
        request.model,
        request.custom_llm_provider,
        request.optional_params,
    );

    PreparedOcrCall {
        request: PreparedOcrRequest {
            config: prepared_provider.config,
            model: prepared_provider.model,
            custom_llm_provider: prepared_provider.custom_llm_provider,
            litellm_call_id: call_id,
            document: request.document,
            api_key: request.api_key.map(str::to_string),
            api_base: request.api_base.map(str::to_string),
            extra_headers: request.extra_headers,
            optional_params: prepared_provider.optional_params,
            auth_inputs: prepared_provider.auth_inputs,
            timeout: request.timeout,
        },
        hooks: OcrLifecycleHooks::new(
            CustomLoggerRunner::new(request.callbacks),
            CustomGuardrailRunner::new(request.guardrails),
            request.request_metadata,
        ),
    }
}

fn new_ocr_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("ocr-{timestamp}-{sequence}")
}

#[cfg(test)]
mod tests {
    use litellm_core::error::{AuthError, Error};
    use serde_json::{Map, json};

    use super::{OcrRequest, prepare_ocr_call};
    use crate::integrations::types::RequestMetadata;

    fn base_ocr_request(model: &str) -> OcrRequest<'_> {
        OcrRequest {
            model,
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-test"),
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Map::new(),
            timeout: None,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: RequestMetadata::default(),
            litellm_call_id: None,
        }
    }

    fn request_with_format(format: &str) -> OcrRequest<'_> {
        let mut request = base_ocr_request("mistral/mistral-ocr-latest");
        request.optional_params = Map::from_iter([("req_format".to_string(), json!(format))]);
        request
    }

    #[test]
    fn native_format_rejected_for_provider_without_support_as_bad_request() {
        let prepared = prepare_ocr_call(request_with_format("native"));
        assert!(
            matches!(prepared.request.config, Err(Error::InvalidRequest(message)) if message.contains("not supported for provider"))
        );
    }

    #[test]
    fn unknown_format_rejected_for_provider_without_support_as_bad_request() {
        let prepared = prepare_ocr_call(request_with_format("raw"));
        assert!(
            matches!(prepared.request.config, Err(Error::InvalidRequest(message)) if message.contains("Invalid `req_format`"))
        );
    }

    #[test]
    fn provider_auth_input_errors_are_rejected_during_preparation() {
        let mut request = base_ocr_request("azure_ai/pixtral-12b-2409");
        request.optional_params =
            Map::from_iter([("tenant_id".to_string(), json!({"invalid": true}))]);

        let prepared = prepare_ocr_call(request);

        assert!(
            matches!(prepared.request.config, Err(Error::Auth(AuthError::InvalidConfiguration(message))) if message.contains("tenant_id"))
        );
    }
}
