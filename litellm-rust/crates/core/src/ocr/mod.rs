pub(crate) mod client;
pub mod error;
mod handler;
pub mod hooks;
pub mod prepare;
pub mod transformation;
pub mod types;
pub mod wire;

pub use types::{OcrRequest, OcrResponseData};

#[tracing::instrument(
    name = "ocr",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn perform_ocr(request: OcrRequest) -> Result<OcrResponseData, crate::Error> {
    handler::perform_ocr_request(request).await
}

#[cfg(test)]
pub(crate) mod tests;
