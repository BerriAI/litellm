use std::sync::OnceLock;
use std::time::Duration;

const HTTP_CLIENT_TIMEOUT_SECS: u64 = 600;
const HTTP_CLIENT_POOL_MAX_IDLE_PER_HOST: usize = 64;
const HTTP_CLIENT_TCP_KEEPALIVE_SECS: u64 = 60;

pub(crate) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_CLIENT_TIMEOUT_SECS))
            .pool_max_idle_per_host(HTTP_CLIENT_POOL_MAX_IDLE_PER_HOST)
            .tcp_keepalive(Some(Duration::from_secs(HTTP_CLIENT_TCP_KEEPALIVE_SECS)))
            .build()
            .expect("failed to build reqwest client")
    })
}
