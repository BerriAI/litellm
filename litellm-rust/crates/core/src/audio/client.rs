//! HTTP client for audio requests.

use std::sync::OnceLock;

use reqwest::Client;

static HTTP_CLIENT: OnceLock<Client> = OnceLock::new();

/// Get the shared HTTP client for audio requests.
pub fn get_http_client() -> &'static Client {
    HTTP_CLIENT.get_or_init(|| {
        Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client")
    })
}
