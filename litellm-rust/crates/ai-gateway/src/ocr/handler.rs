use litellm_core::error::Error;
use serde_json::Value;

use super::hooks::OcrLifecycleHooks;
use super::types::PreparedOcrRequest;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(
    request: PreparedOcrRequest,
    hooks: &OcrLifecycleHooks,
) -> Result<Value, Error> {
    let request = hooks.prepare_provider_request(request).await?;
    litellm_core::ocr::execute_ocr_provider_call(request).await
}
