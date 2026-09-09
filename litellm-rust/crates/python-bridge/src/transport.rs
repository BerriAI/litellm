use std::sync::OnceLock;

use litellm_core::error::{Error, TransportError};
use litellm_core::ocr::OcrClient;

pub(crate) fn ocr_client() -> Result<OcrClient, Error> {
    static CLIENT: OnceLock<Result<OcrClient, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .build()
                .map_err(|error| error.to_string())
                .and_then(|provider_http| {
                    OcrClient::new(provider_http).map_err(|error| error.to_string())
                })
        })
        .as_ref()
        .cloned()
        .map_err(|error| TransportError::Network(error.clone()).into())
}
