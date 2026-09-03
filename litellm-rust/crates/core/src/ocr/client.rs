use std::sync::OnceLock;
use std::time::Duration;

use crate::Error;
use crate::constants::{MESSAGES_CONNECT_TIMEOUT_SECS, OCR_TIMEOUT_SECS};

pub(super) fn http_client() -> Result<&'static reqwest::Client, Error> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(MESSAGES_CONNECT_TIMEOUT_SECS))
                .timeout(Duration::from_secs(OCR_TIMEOUT_SECS))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| Error::Network(error.clone()))
}
