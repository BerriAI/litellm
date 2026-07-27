use std::sync::OnceLock;
use std::time::Duration;

use crate::constants::DEFAULT_HTTP_CLIENT_TIMEOUT_SECS;

pub(crate) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_HTTP_CLIENT_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}

pub(crate) fn safe_fetch_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(DEFAULT_HTTP_CLIENT_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}
