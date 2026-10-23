use std::{fmt, io, path::PathBuf, sync::Arc};

use tokio_tungstenite::tungstenite;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsSource {
    CaBundle,
    ClientIdentity,
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(Arc<Repr>);

#[derive(Debug, thiserror::Error)]
pub(crate) enum Repr {
    #[error("could not read {}: {cause}", path.display())]
    Read {
        path: PathBuf,
        tls_source: TlsSource,
        cause: io::Error,
    },
    #[error("{} is not a PEM file: {cause}", path.display())]
    InvalidPem {
        path: PathBuf,
        tls_source: TlsSource,
        cause: PemFault,
    },
    #[error("could not build the HTTP client: {}", Chain(.0))]
    Client(reqwest::Error),
    #[error("could not build the HTTP client: {0}")]
    TlsConfig(rustls::Error),

    #[error("request body could not be serialized: {0}")]
    RequestBody(serde_json::Error),
    #[error("invalid request: {0}")]
    Request(http::Error),
    #[error("invalid request: {}", Chain(.0))]
    RequestUrl(reqwest::Error),
    #[error("request forwards a header the signer computes: {0}")]
    ComputedHeader(String),
    #[error("request signing failed: a streaming body cannot be signed")]
    UnsignableBody,
    #[error("request signing failed: {0}")]
    Signature(Box<dyn std::error::Error + Send + Sync>),

    #[error("could not reach the provider: {}", Chain(.0))]
    Connect(reqwest::Error),
    #[error("upstream network error: {}", Chain(.0))]
    Network(reqwest::Error),
    #[error("upstream request timed out: {}", Chain(.0))]
    Timeout(reqwest::Error),
    #[error("upstream request failed with status {status}: {body}")]
    Status { status: u16, body: String },

    #[error("invalid WebSocket URL: {0}")]
    WebSocketUrl(url::ParseError),
    #[error("unsupported WebSocket URL scheme {0:?}")]
    WebSocketScheme(String),
    #[error("WebSocket handshake returned a wrong Sec-WebSocket-Accept")]
    HandshakeAccept,
    #[error("WebSocket handshake returned an invalid Upgrade header")]
    HandshakeUpgrade,
    #[error("WebSocket handshake returned an invalid Connection header")]
    HandshakeConnection,
    #[error("WebSocket handshake returned an invalid subprotocol")]
    HandshakeSubprotocol,
    #[error("WebSocket connection timed out")]
    HandshakeTimeout,
    #[error("upstream WebSocket error: {0}")]
    WebSocket(tungstenite::Error),

    #[error("media URL rejected by network policy")]
    BlockedUrl,
    #[error("could not resolve the media host: {0}")]
    Resolve(io::Error),
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
    DownloadStatus(u16),
    #[error("media download timed out")]
    DownloadTimeout,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PemFault {
    #[error("no certificates found")]
    NoCertificates,
    #[error(transparent)]
    Parse(#[from] rustls::pki_types::pem::Error),
    #[error(transparent)]
    Rejected(#[from] rustls::Error),
}

struct Chain<'a>(&'a reqwest::Error);

impl fmt::Display for Chain<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)?;
        std::iter::successors(std::error::Error::source(self.0), |cause| cause.source())
            .try_for_each(|cause| write!(formatter, ": {cause}"))
    }
}

impl Error {
    pub fn upstream(status: u16, body: String) -> Self {
        Repr::Status { status, body }.into()
    }

    pub fn computed_header(name: impl Into<String>) -> Self {
        Repr::ComputedHeader(name.into()).into()
    }

    pub fn unsignable_body() -> Self {
        Repr::UnsignableBody.into()
    }

    pub fn signature(cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Repr::Signature(Box::new(cause)).into()
    }

    /// The request may have reached the provider, so retrying elsewhere could
    /// repeat a billed call.
    pub fn dispatched(&self) -> bool {
        matches!(
            *self.0,
            Repr::Network(_)
                | Repr::Timeout(_)
                | Repr::Status { .. }
                | Repr::HandshakeAccept
                | Repr::HandshakeUpgrade
                | Repr::HandshakeConnection
                | Repr::HandshakeSubprotocol
                | Repr::HandshakeTimeout
                | Repr::WebSocket(_)
                | Repr::TooManyRedirects
                | Repr::MissingRedirectLocation
                | Repr::InvalidRedirect
                | Repr::DownloadTooLarge
                | Repr::DownloadStatus(_)
                | Repr::DownloadTimeout
        )
    }

    pub fn is_request(&self) -> bool {
        matches!(
            *self.0,
            Repr::Request(_)
                | Repr::RequestUrl(_)
                | Repr::ComputedHeader(_)
                | Repr::WebSocketUrl(_)
                | Repr::WebSocketScheme(_)
                | Repr::BlockedUrl
                | Repr::DownloadDisabled
                | Repr::DownloadTooLarge
                | Repr::TooManyRedirects
        )
    }

    pub fn is_response(&self) -> bool {
        matches!(
            *self.0,
            Repr::MissingRedirectLocation | Repr::InvalidRedirect
        )
    }

    pub fn is_computed_header(&self) -> bool {
        matches!(*self.0, Repr::ComputedHeader(_))
    }

    pub fn upstream_response(&self) -> Option<(u16, &str)> {
        match &*self.0 {
            Repr::Status { status, body } => Some((*status, body)),
            _ => None,
        }
    }

    pub fn status(&self) -> Option<u16> {
        match *self.0 {
            Repr::Status { status, .. } | Repr::DownloadStatus(status) => Some(status),
            Repr::Timeout(_) | Repr::DownloadTimeout => Some(408),
            _ => None,
        }
    }

    pub fn tls_source(&self) -> Option<TlsSource> {
        match *self.0 {
            Repr::Read { tls_source, .. } | Repr::InvalidPem { tls_source, .. } => Some(tls_source),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn repr(&self) -> &Repr {
        &self.0
    }
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Error {}

impl From<Repr> for Error {
    fn from(repr: Repr) -> Self {
        Self(Arc::new(repr))
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        let error = error.without_url();
        if error.is_timeout() {
            Repr::Timeout(error).into()
        } else if error.is_connect() || error.is_builder() {
            Repr::Connect(error).into()
        } else {
            Repr::Network(error).into()
        }
    }
}

impl From<tungstenite::Error> for Error {
    fn from(error: tungstenite::Error) -> Self {
        Repr::WebSocket(error).into()
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
        let error = crate::Error::from(error);
        assert!(!error.dispatched());
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
        let message = crate::Error::from(error).to_string();
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
        let error = crate::Error::from(error);
        assert!(error.dispatched());
        assert_eq!(error.status(), Some(408));
    }
}
