use litellm_core::Error;
use litellm_core::ocr::observers::OcrObserver;
use litellm_core::ocr::{PreparedOcrRequest, execute_ocr_provider_call as core_execute};
use serde_json::Value;

use super::hooks::OcrLifecycleHooks;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(
    request: PreparedOcrRequest,
    hooks: &OcrLifecycleHooks,
    observer: &mut impl OcrObserver,
) -> Result<Value, Error> {
    let request = hooks.prepare_provider_request(request).await?;
    core_execute(request, observer).await
}
