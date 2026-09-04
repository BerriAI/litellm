mod common_utils;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::Error;
use handler::execute_ocr_provider_call;
use prepare::prepare_ocr_call;
pub use types::OcrRequest;

/// The injected client must disable automatic redirects so document URLs are
/// validated before each redirect is followed
#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(client: &reqwest::Client, request: OcrRequest<'_>) -> Result<Value, Error> {
    let provider_request = prepare_ocr_call(request).await?;
    execute_ocr_provider_call(client, provider_request).await
}

#[cfg(test)]
mod tests;
