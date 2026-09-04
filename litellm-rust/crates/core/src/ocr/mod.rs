mod common_utils;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::Error;
pub use types::{OcrRequest, PreparedOcrRequest};

/// The injected client must disable automatic redirects so document URLs are
/// validated before each redirect is followed
#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(client: &reqwest::Client, request: OcrRequest<'_>) -> Result<Value, Error> {
    let provider_request = prepare(request).await?;
    execute(client, provider_request).await
}

pub async fn prepare(request: OcrRequest<'_>) -> Result<PreparedOcrRequest, Error> {
    prepare::prepare_ocr_call(request).await
}

pub async fn execute(
    client: &reqwest::Client,
    request: PreparedOcrRequest,
) -> Result<Value, Error> {
    handler::execute_ocr_provider_call(client, request).await
}

#[cfg(test)]
mod tests;
