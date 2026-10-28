use std::{borrow::Cow, fmt, io, path::PathBuf, sync::Arc};

use tokio::time::error::Elapsed;
use tokio_tungstenite::tungstenite;

const UPSTREAM_BODY_MAX_CHARS: usize = 256;
const TRUNCATION_MARKER: &str = "... (truncated)";

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
    #[error("could not read {}", path.display())]
    Read {
        path: PathBuf,
        tls_source: TlsSource,
        source: io::Error,
    },
    #[error("could not load {}", path.display())]
    InvalidPem {
        path: PathBuf,
        tls_source: TlsSource,
        source: PemFault,
    },
    #[error("could not build the HTTP client")]
    Client(#[source] reqwest::Error),
    #[error("could not build the TLS configuration")]
    TlsConfig(#[source] rustls::Error),

    #[error("request body could not be serialized")]
    RequestBody(#[source] serde_json::Error),
    #[error("invalid request")]
    Request(#[source] http::Error),
    #[error("invalid request")]
    RequestUrl(#[source] reqwest::Error),
    #[error("invalid request URL")]
    ParseUrl(#[source] url::ParseError),
    #[error("request forwards the header {0:?}, which the signer computes")]
    ComputedHeader(String),
    #[error("request signing failed: a streaming body cannot be signed")]
    UnsignableBody,
    #[error("request forwards a header that cannot be signed")]
    UnsignableHeader(#[source] litellm_error::Source),
    #[error("request signing failed")]
    Signature(#[source] litellm_error::Source),

    #[error("could not reach the provider")]
    Connect(#[source] reqwest::Error),
    #[error("upstream network error")]
    Network(#[source] reqwest::Error),
    #[error("upstream request timed out")]
    Timeout(#[source] reqwest::Error),
    #[error("upstream request failed with status {status}: {body}")]
    Status { status: u16, body: UpstreamBody },

    #[error("invalid WebSocket URL")]
    WebSocketUrl(#[source] url::ParseError),
    #[error("unsupported WebSocket URL scheme {0:?}")]
    WebSocketScheme(String),
    #[error("could not rewrite the WebSocket URL scheme to {0}")]
    WebSocketSchemeRewrite(&'static str),
    #[error("WebSocket handshake returned {0}")]
    Handshake(HandshakeFault),
    #[error("WebSocket connection timed out")]
    HandshakeTimeout(#[source] Elapsed),
    #[error("upstream WebSocket error")]
    WebSocket(#[source] tungstenite::Error),
    #[error("the WebSocket connection is closed")]
    WebSocketClosed,

    #[error("media URL rejected by network policy")]
    BlockedUrl,
    #[error("could not resolve the media host")]
    Resolve(#[source] io::Error),
    #[error("media download is disabled")]
    DownloadDisabled,
    #[error("media download exceeds the maximum size")]
    DownloadTooLarge,
    #[error("too many redirects while fetching media")]
    TooManyRedirects,
    #[error("media redirect is missing a Location header")]
    MissingRedirectLocation,
    #[error("invalid media redirect")]
    InvalidRedirect(#[source] RedirectFault),
    #[error("media download failed with status {0}")]
    DownloadStatus(u16),
    #[error("media download failed")]
    Download(#[source] reqwest::Error),
    #[error("media download timed out")]
    DownloadTimeout(#[source] Elapsed),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HandshakeFault {
    #[error("a wrong Sec-WebSocket-Accept")]
    Accept,
    #[error("an invalid Upgrade header")]
    Upgrade,
    #[error("an invalid Connection header")]
    Connection,
    #[error("an invalid subprotocol")]
    Subprotocol,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PemFault {
    #[error("no certificates found")]
    NoCertificates,
    #[error("could not parse PEM")]
    Parse(#[from] rustls::pki_types::pem::Error),
    #[error("rustls rejected the certificate or key")]
    Rejected(#[from] rustls::Error),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RedirectFault {
    #[error("Location header is not valid text")]
    Location(#[from] http::header::ToStrError),
    #[error("Location is not a valid URL")]
    Url(#[from] url::ParseError),
}

impl Repr {
    pub(crate) fn download(error: reqwest::Error) -> Self {
        Self::Download(error.without_url())
    }
}

impl Error {
    pub fn upstream(status: u16, body: String) -> Self {
        Repr::Status {
            status,
            body: UpstreamBody(body),
        }
        .into()
    }

    pub fn computed_header(name: impl Into<String>) -> Self {
        Repr::ComputedHeader(name.into()).into()
    }

    pub fn unsignable_body() -> Self {
        Repr::UnsignableBody.into()
    }

    pub fn unsignable_header(cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Repr::UnsignableHeader(litellm_error::Source::new(cause)).into()
    }

    pub fn websocket_closed() -> Self {
        Repr::WebSocketClosed.into()
    }

    pub fn signature(cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Repr::Signature(litellm_error::Source::new(cause)).into()
    }

    /// The request may have reached the provider, so retrying elsewhere could
    /// repeat a billed call.
    pub fn dispatched(&self) -> bool {
        match &*self.0 {
            Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::Status { .. }
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_) => true,
            Repr::WebSocketClosed
            | Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => false,
        }
    }

    pub fn is_request(&self) -> bool {
        match &*self.0 {
            Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableHeader(_)
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::BlockedUrl
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects => true,
            Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::UnsignableBody
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::Status { .. }
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::Resolve(_)
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => false,
        }
    }

    pub fn is_response(&self) -> bool {
        match &*self.0 {
            Repr::MissingRedirectLocation | Repr::InvalidRedirect(_) => true,
            Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::Status { .. }
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => false,
        }
    }

    pub fn is_computed_header(&self) -> bool {
        match &*self.0 {
            Repr::ComputedHeader(_) => true,
            Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::Status { .. }
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => false,
        }
    }

    pub fn is_connect(&self) -> bool {
        match &*self.0 {
            Repr::Connect(_) => true,
            Repr::ComputedHeader(_)
            | Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::Status { .. }
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => false,
        }
    }

    pub fn is_timeout(&self) -> bool {
        match &*self.0 {
            Repr::Timeout(_) | Repr::HandshakeTimeout(_) | Repr::DownloadTimeout(_) => true,
            Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::Status { .. }
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_) => false,
        }
    }

    pub fn upstream_response(&self) -> Option<(u16, &str)> {
        match &*self.0 {
            Repr::Status { status, body } => Some((*status, &body.0)),
            Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => None,
        }
    }

    pub fn status(&self) -> Option<u16> {
        match &*self.0 {
            Repr::Status { status, .. } | Repr::DownloadStatus(status) => Some(*status),
            Repr::Timeout(_) | Repr::DownloadTimeout(_) => Some(408),
            Repr::Read { .. }
            | Repr::InvalidPem { .. }
            | Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::Download(_) => None,
        }
    }

    pub fn tls_source(&self) -> Option<TlsSource> {
        match &*self.0 {
            Repr::Read { tls_source, .. } | Repr::InvalidPem { tls_source, .. } => {
                Some(*tls_source)
            }
            Repr::Client(_)
            | Repr::TlsConfig(_)
            | Repr::RequestBody(_)
            | Repr::Request(_)
            | Repr::RequestUrl(_)
            | Repr::ParseUrl(_)
            | Repr::ComputedHeader(_)
            | Repr::UnsignableBody
            | Repr::UnsignableHeader(_)
            | Repr::Signature(_)
            | Repr::Connect(_)
            | Repr::Network(_)
            | Repr::Timeout(_)
            | Repr::Status { .. }
            | Repr::WebSocketUrl(_)
            | Repr::WebSocketScheme(_)
            | Repr::WebSocketSchemeRewrite(_)
            | Repr::Handshake(_)
            | Repr::HandshakeTimeout(_)
            | Repr::WebSocket(_)
            | Repr::WebSocketClosed
            | Repr::BlockedUrl
            | Repr::Resolve(_)
            | Repr::DownloadDisabled
            | Repr::DownloadTooLarge
            | Repr::TooManyRedirects
            | Repr::MissingRedirectLocation
            | Repr::InvalidRedirect(_)
            | Repr::DownloadStatus(_)
            | Repr::Download(_)
            | Repr::DownloadTimeout(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn repr(&self) -> &Repr {
        &self.0
    }
}

pub(crate) struct UpstreamBody(String);

impl fmt::Display for UpstreamBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&bounded(&self.0), f)
    }
}

impl fmt::Debug for UpstreamBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&bounded(&self.0), f)
    }
}

fn bounded(body: &str) -> Cow<'_, str> {
    match body.char_indices().nth(UPSTREAM_BODY_MAX_CHARS) {
        None => Cow::Borrowed(body),
        Some((cut, _)) => Cow::Owned(format!("{}{TRUNCATION_MARKER}", &body[..cut])),
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
    use litellm_error::Report;
    use rstest::rstest;

    use super::{TRUNCATION_MARKER, UPSTREAM_BODY_MAX_CHARS};

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
        let report = Report::new(&error).to_string();
        assert!(!error.dispatched());
        assert!(!report.contains("secret"), "{report}");
        assert!(!report.contains("private"), "{report}");
    }

    fn root_cause<'a>(
        error: &'a (dyn std::error::Error + 'static),
    ) -> &'a (dyn std::error::Error + 'static) {
        error.source().map_or(error, root_cause)
    }

    #[tokio::test]
    async fn connect_failure_reports_its_root_cause_once() {
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
        let error = crate::Error::from(error);
        let root_cause = root_cause(&error).to_string();
        let report = Report::new(&error).to_string();
        assert_eq!(report.matches(&root_cause).count(), 1, "{report}");
        assert!(!error.to_string().contains(&root_cause));
        assert!(error.is_connect());
        assert!(!error.is_timeout());
        assert!(!report.contains("secret"), "{report}");
        assert!(!report.contains("private"), "{report}");
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
        assert!(error.is_timeout());
        assert!(!error.is_connect());
        assert_eq!(error.status(), Some(408));
    }

    #[rstest]
    #[case::empty(String::new(), String::new())]
    #[case::short("slow down".to_owned(), "slow down".to_owned())]
    #[case::ascii_over_the_limit(
        format!("{}private-tail", "a".repeat(UPSTREAM_BODY_MAX_CHARS)),
        format!("{}{TRUNCATION_MARKER}", "a".repeat(UPSTREAM_BODY_MAX_CHARS))
    )]
    #[case::at_the_limit("\u{00e9}".repeat(UPSTREAM_BODY_MAX_CHARS), "\u{00e9}".repeat(UPSTREAM_BODY_MAX_CHARS))]
    #[case::over_the_limit(
        format!("{}private-tail", "\u{00e9}".repeat(UPSTREAM_BODY_MAX_CHARS)),
        format!("{}{TRUNCATION_MARKER}", "\u{00e9}".repeat(UPSTREAM_BODY_MAX_CHARS))
    )]
    fn upstream_preserves_the_body_and_bounds_only_diagnostics(
        #[case] body: String,
        #[case] expected: String,
    ) {
        let error = crate::Error::upstream(429, body.clone());
        let expected_message = format!("upstream request failed with status 429: {expected}");
        assert_eq!(error.to_string(), expected_message);
        assert_eq!(Report::new(&error).to_string(), expected_message);
        let debug = format!("{error:?}");
        assert!(debug.contains(&format!("{expected:?}")), "{debug}");
        assert!(!debug.contains("private-tail"), "{debug}");
        assert!(!format!("{error:#?}").contains("private-tail"));
        assert_eq!(error.upstream_response(), Some((429, body.as_str())));
        assert_eq!(error.status(), Some(429));
    }
}
