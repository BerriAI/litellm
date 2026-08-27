use std::sync::OnceLock;
use std::time::Duration;

use crate::constants::{MESSAGES_CONNECT_TIMEOUT_SECS, MESSAGES_TIMEOUT_SECS};

pub(super) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(MESSAGES_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(MESSAGES_CONNECT_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}
