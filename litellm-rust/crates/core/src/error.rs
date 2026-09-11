use thiserror::Error as ThisError;

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum Error {
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
    #[error("{0}")]
    Auth(String),
    #[error(
        "Missing {provider} API Key - A call is being made to {provider} but no key is set either in the environment variables or via params"
    )]
    MissingApiKey { provider: &'static str },
    #[error(
        "invalid authentication configuration: Missing Azure AI credentials - set AZURE_AI_API_KEY or configure Entra ID"
    )]
    MissingAzureAiCredentials,
    #[error("Missing Azure AI credentials - set AZURE_AI_API_KEY or provide azure_ad_token")]
    MissingAzureAiCredentialsOrAdToken,
    #[error(
        "invalid authentication configuration: Missing Azure Document Intelligence credentials - set AZURE_DOCUMENT_INTELLIGENCE_API_KEY or configure Entra ID"
    )]
    MissingAzureDocumentIntelligenceCredentials,
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

impl From<reqwest::Error> for TransportError {
    fn from(error: reqwest::Error) -> Self {
        Self::Network(error.without_url().to_string())
    }
}

impl From<crate::ocr::error::OcrRequestError> for Error {
    fn from(error: crate::ocr::error::OcrRequestError) -> Self {
        match error {
            crate::ocr::error::OcrRequestError::MissingField(field) => Self::MissingField(field),
            error => Self::InvalidRequest(error.to_string()),
        }
    }
}

impl From<crate::ocr::error::OcrResponseError> for Error {
    fn from(error: crate::ocr::error::OcrResponseError) -> Self {
        Self::InvalidResponse(error.to_string())
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

impl From<crate::AuthError> for Error {
    fn from(error: crate::AuthError) -> Self {
        match error {
            crate::AuthError::MissingApiKey { provider } => Self::MissingApiKey { provider },
            error => Self::Auth(error.to_string()),
        }
    }
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

#[cfg(test)]
mod transport_tests {
    use super::*;

    #[test]
    fn missing_auth_key_preserves_provider_in_public_error() {
        assert_eq!(
            Error::from(crate::AuthError::MissingApiKey { provider: "Vertex" }),
            Error::MissingApiKey { provider: "Vertex" }
        );
    }

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
        let error = TransportError::from_reqwest_before_dispatch(error);
        assert!(matches!(error, TransportError::Connect(_)));
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
            TransportError::from_reqwest_before_dispatch(error),
            TransportError::Network(_)
        ));
    }
}
