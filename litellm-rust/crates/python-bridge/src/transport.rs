use std::sync::OnceLock;

use litellm_core::error::{Error, TransportError};
use litellm_core::ocr::OcrClient;

pub(crate) fn ocr_client() -> Result<OcrClient, Error> {
    static CLIENT: OnceLock<Result<OcrClient, TransportError>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .build()
                .map_err(TransportError::from)
                .and_then(OcrClient::new)
        })
        .clone()
        .map_err(Error::from)
}
