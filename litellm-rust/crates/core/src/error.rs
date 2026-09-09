use thiserror::Error as ThisError;

pub use crate::auth::error::AuthError;

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("invalid request: {0}")]
    OcrRequest(#[from] crate::ocr::error::OcrRequestError),
    #[error("invalid response: {0}")]
    OcrResponse(#[from] crate::ocr::error::OcrResponseError),
    #[error("{0}")]
    OcrPolling(#[from] crate::ocr::error::OcrPollingError),
    #[error("invalid request: {0}")]
    ChatRequest(#[from] crate::chat_completions::error::ChatRequestError),
    #[error("invalid response: {0}")]
    ChatResponse(#[from] crate::chat_completions::error::ChatResponseError),
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Auth(#[from] AuthError),
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    /// The provider was never reached: DNS, TCP, TLS or proxy setup failed
    /// before any byte of the request went out. Nothing was billed, so a host
    /// that keeps a reference implementation can serve the request itself.
    /// A timeout is deliberately not this, since the provider may have received
    /// and answered the request already.
    #[error("could not reach the provider: {0}")]
    Connect(String),
    #[error("routing error: {0}")]
    Routing(String),
    /// The request is outside the surface this route covers in Rust. Hosts that
    /// keep a reference implementation treat this as "fall back", not "fail".
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
}

pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[derive(Debug, ThisError)]
pub(crate) enum MediaError {
    #[error("media URL rejected by network policy")]
    BlockedUrl,
    #[error("media download is disabled")]
    DownloadDisabled,
    #[error("media download exceeds the maximum size")]
    DownloadTooLarge,
    #[error("too many redirects while fetching media")]
    TooManyRedirects,
    #[error("media redirect is missing a Location header")]
    MissingRedirectLocation,
    #[error("invalid media redirect")]
    InvalidRedirect,
    #[error("media download failed with status {0}")]
    Http(u16),
    #[error("media download timed out")]
    Timeout,
    #[error("{0}")]
    Transport(#[from] TransportError),
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum TransportError {
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    #[error("could not reach the provider: {0}")]
    Connect(String),
}

impl TransportError {
    pub fn before_request(error: reqwest::Error) -> Self {
        let before_dispatch = !error.is_timeout() && (error.is_connect() || error.is_builder());
        let message = error.without_url().to_string();
        if before_dispatch {
            Self::Connect(message)
        } else {
            Self::Network(message)
        }
    }
}

impl From<reqwest::Error> for TransportError {
    fn from(error: reqwest::Error) -> Self {
        Self::Network(error.without_url().to_string())
    }
}

impl From<TransportError> for Error {
    fn from(error: TransportError) -> Self {
        match error {
            TransportError::Http { status, body } => Self::Http { status, body },
            TransportError::Network(message) => Self::Network(message),
            TransportError::Connect(message) => Self::Connect(message),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidType,
    MissingField,
    InvalidRequest,
    InvalidResponse,
    InvalidProvider,
    Auth,
    Http,
    Network,
    Connect,
    Routing,
    Unsupported,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        use crate::ocr::error::{OcrPollingError, OcrRequestError};
        match self {
            Self::InvalidType { .. } => ErrorKind::InvalidType,
            Self::MissingField(_) | Self::OcrRequest(OcrRequestError::MissingField(_)) => {
                ErrorKind::MissingField
            }
            Self::InvalidRequest(_) | Self::OcrRequest(_) | Self::ChatRequest(_) => {
                ErrorKind::InvalidRequest
            }
            Self::InvalidResponse(_) | Self::OcrResponse(_) | Self::ChatResponse(_) => {
                ErrorKind::InvalidResponse
            }
            Self::InvalidProvider(_) => ErrorKind::InvalidProvider,
            Self::Auth(_) => ErrorKind::Auth,
            Self::Http { .. } => ErrorKind::Http,
            Self::Network(_) | Self::OcrPolling(OcrPollingError::PollTimeout) => ErrorKind::Network,
            Self::OcrPolling(OcrPollingError::PollOrigin | OcrPollingError::PollLocation) => {
                ErrorKind::InvalidResponse
            }
            Self::Connect(_) => ErrorKind::Connect,
            Self::Routing(_) => ErrorKind::Routing,
            Self::Unsupported(_) => ErrorKind::Unsupported,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let error = TransportError::before_request(error);
        assert!(matches!(error, TransportError::Connect(_)));
        assert!(!error.to_string().contains("secret"));
        assert!(!error.to_string().contains("private"));

        let error = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .get("http://localhost:invalid/private?api_key=secret")
            .send()
            .await
            .expect_err("invalid port");
        let error = TransportError::from(error);
        assert!(matches!(error, TransportError::Network(_)));
        assert!(!error.to_string().contains("secret"));
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
            TransportError::before_request(error),
            TransportError::Network(_)
        ));
    }
}
