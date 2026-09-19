#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    #[error("could not reach the provider: {0}")]
    Connect(String),
}

impl Error {
    pub fn from_reqwest_before_dispatch(error: reqwest::Error) -> Self {
        let before_dispatch = !error.is_timeout() && (error.is_connect() || error.is_builder());
        let message = error.without_url().to_string();
        if before_dispatch {
            Self::Connect(message)
        } else {
            Self::Network(message)
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Network(error.without_url().to_string())
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn transport_errors_remove_urls_and_keep_dispatch_context() {
        let error = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get("http://localhost:invalid/private?api_key=secret")
            .send()
            .await
            .expect_err("invalid port");
        let error = crate::custom_httpx::transport::Error::from_reqwest_before_dispatch(error);
        assert!(matches!(
            error,
            crate::custom_httpx::transport::Error::Connect(_)
        ));
        assert!(!error.to_string().contains("secret"));
        assert!(!error.to_string().contains("private"));
    }

    #[tokio::test]
    async fn request_timeout_is_not_safe_to_retry_as_a_connect_failure() {
        use std::time::Duration;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let request = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(format!("http://{address}"))
            .timeout(Duration::from_millis(200))
            .send();
        let (response, accepted) = tokio::join!(
            request,
            tokio::time::timeout(Duration::from_secs(2), listener.accept())
        );
        let _connection = accepted
            .expect("accept deadline")
            .expect("accepted connection");
        let error = response.expect_err("server does not respond");
        assert!(error.is_timeout());
        assert!(matches!(
            crate::custom_httpx::transport::Error::from_reqwest_before_dispatch(error),
            crate::custom_httpx::transport::Error::Network(_)
        ));
    }
}
