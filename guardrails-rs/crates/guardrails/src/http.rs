use std::time::Duration;

pub struct HttpClient {
    inner: reqwest::Client,
}

impl HttpClient {
    pub fn new(default_timeout: Duration) -> Self {
        let inner = reqwest::Client::builder()
            .timeout(default_timeout)
            .pool_max_idle_per_host(32)
            .build()
            .expect("failed to build reqwest client");
        Self { inner }
    }

    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }
}
