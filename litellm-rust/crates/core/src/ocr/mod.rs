mod client;
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

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(request: OcrRequest<'_>) -> Result<Value, Error> {
    let provider_request = prepare_ocr_call(request).await?;
    execute_ocr_provider_call(provider_request).await
}

#[cfg(test)]
mod tests;
