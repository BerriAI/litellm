use litellm_core::Error;
use litellm_core::call_lifecycle::CallLifecycle;
use serde_json::Value;

mod common_utils;
mod handler;
mod hooks;
mod prepare;
mod types;

pub use types::OcrRequest;

use handler::execute_ocr_provider_call;
use prepare::{PreparedOcrCall, prepare_ocr_call};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(request: OcrRequest<'_>) -> Result<Value, Error> {
    let PreparedOcrCall { request, hooks } = prepare_ocr_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, |request| {
            execute_ocr_provider_call(request, &hooks)
        })
        .await
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
        assert!(!is_supported_request("deepseek-ocr", Some("vertex_ai")));
    }
}
