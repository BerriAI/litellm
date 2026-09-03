use litellm_core::Error;
use litellm_core::call_lifecycle::CallLifecycle;
use litellm_core::ocr::observers::{NoopOcrObserver, OcrObserver};
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
    ocr_with_observer(request, &mut NoopOcrObserver).await
}

#[tracing::instrument(
    name = "ocr",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn ocr_with_observer(
    request: OcrRequest<'_>,
    observer: &mut impl OcrObserver,
) -> Result<Value, Error> {
    let PreparedOcrCall { request, hooks } = prepare_ocr_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, |request| {
            execute_ocr_provider_call(request, &hooks, observer)
        })
        .await
}

#[cfg(test)]
mod tests;
