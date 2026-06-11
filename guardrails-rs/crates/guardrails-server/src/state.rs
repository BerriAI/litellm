use std::sync::Arc;
use std::time::Duration;

use guardrails_http::HttpClient;

#[derive(Clone)]
pub struct AppState {
    pub http: Arc<HttpClient>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            http: Arc::new(HttpClient::new(Duration::from_secs(10))),
        }
    }
}
