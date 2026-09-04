use std::sync::OnceLock;
use std::time::Duration;

use crate::constants::{HTTP_CLIENT_CONNECT_TIMEOUT_SECS, HTTP_CLIENT_TIMEOUT_SECS};

pub(crate) fn shared_http_client() -> Result<reqwest::Client, String> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_CLIENT_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(HTTP_CLIENT_CONNECT_TIMEOUT_SECS))
            // OCR validates each redirect before fetching it
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| error.to_string())
    });
    client.clone()
}
