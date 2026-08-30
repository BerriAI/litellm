use std::sync::OnceLock;
use std::time::Duration;

use crate::constants::{HTTP_CLIENT_TIMEOUT_SECS, HTTP_CONNECT_TIMEOUT_SECS};

pub(super) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(HTTP_CLIENT_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}
