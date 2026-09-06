mod client;
mod common_utils;
pub mod handler;
pub mod observers;
pub mod prepare;
pub mod transformation;
pub mod types;

pub use handler::execute_ocr_provider_call;
pub use prepare::{prepare_ocr_call, prepare_ocr_provider_call};
pub use types::{OcrRequest, PreparedOcrRequest, ProviderOcrRequest};

use serde_json::Value;

use crate::Error;
use crate::ocr::observers::{NoopOcrObserver, OcrObserver};

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
    let prepared = prepare_ocr_call(request);
    let provider_request = prepare_ocr_provider_call(prepared).await?;
    execute_ocr_provider_call(provider_request, observer).await
}

pub fn ocr_admitted(model: &str, provider: &str, request_format: Option<&str>) -> bool {
    prepare::ocr_provider_config(provider, model).is_some_and(|config| {
        request_format != Some("native") || config.supported_ocr_params().contains(&"req_format")
    })
}

#[cfg(test)]
mod tests;
