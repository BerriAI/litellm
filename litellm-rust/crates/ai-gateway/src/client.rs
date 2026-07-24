use std::sync::OnceLock;
use std::time::Duration;

const HTTP_CLIENT_TIMEOUT_SECS: u64 = 600;

pub(crate) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_CLIENT_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}
