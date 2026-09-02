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
        .run_request(request, &hooks, execute_ocr_provider_call)
        .await
}

#[cfg(test)]
mod tests;
