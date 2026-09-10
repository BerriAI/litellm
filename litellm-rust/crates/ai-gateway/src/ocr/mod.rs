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

#[cfg(test)]
mod tests {
    use litellm_core::ocr::wire::is_supported_request;

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
}
