pub(crate) mod client;
pub(crate) mod document;
pub mod error;
mod handler;
pub mod hooks;
pub mod prepare;
pub mod transformation;
pub mod types;
pub mod wire;

pub use types::{OcrRequest, OcrResponseData};

/// Execute an OCR request with a host-owned HTTP client.
///
/// The client must not follow redirects automatically. OCR document downloads
/// validate each redirect target before issuing the next request.
#[tracing::instrument(
    name = "ocr",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn perform_ocr(
    http_client: &reqwest::Client,
    request: OcrRequest,
) -> Result<OcrResponseData, crate::Error> {
    handler::perform_ocr_request(http_client, request).await
}

#[cfg(test)]
#[path = "../../tests/ocr.rs"]
pub(crate) mod tests;
