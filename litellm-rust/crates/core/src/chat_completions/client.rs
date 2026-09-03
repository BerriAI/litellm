use std::sync::OnceLock;
use std::time::Duration;

use crate::constants::{
    CHAT_COMPLETIONS_CONNECT_TIMEOUT_SECS, CHAT_COMPLETIONS_TIMEOUT_SECS,
    PROVIDER_POOL_MAX_IDLE_PER_HOST, PROVIDER_TCP_KEEPALIVE_SECS,
};

pub(super) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(CHAT_COMPLETIONS_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(CHAT_COMPLETIONS_CONNECT_TIMEOUT_SECS))
            .pool_max_idle_per_host(PROVIDER_POOL_MAX_IDLE_PER_HOST)
            .tcp_keepalive(Some(Duration::from_secs(PROVIDER_TCP_KEEPALIVE_SECS)))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}
