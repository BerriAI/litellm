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
        let message = describe(error);
        if before_dispatch {
            Self::Connect(message)
        } else {
            Self::Network(message)
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Network(describe(error))
    }
}

fn describe(error: reqwest::Error) -> String {
    let error = error.without_url();
    std::iter::successors(std::error::Error::source(&error), |cause| cause.source())
        .fold(error.to_string(), |message, cause| {
            format!("{message}: {cause}")
        })
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

    fn root_cause(error: &dyn std::error::Error) -> Option<String> {
        match error.source() {
            Some(cause) => root_cause(cause).or_else(|| Some(cause.to_string())),
            None => None,
        }
    }

    #[tokio::test]
    async fn network_error_message_names_the_underlying_cause() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        drop(listener);
        let error = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get(format!("http://{address}/private?api_key=secret"))
            .send()
            .await
            .expect_err("nothing listens on the port");
        let root_cause = root_cause(&error).expect("reqwest reports a cause");
        let message = crate::custom_httpx::transport::Error::from(error).to_string();
        assert!(message.contains(&root_cause), "{message}");
        assert!(!message.contains("secret"));
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
