#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error("CyberArk Conjur operation timed out")]
    Timeout,
    #[error("CyberArk Conjur HTTP request failed")]
    Http(
        #[from]
        #[redact]
        litellm_http::Error,
    ),
    #[error("CyberArk Conjur authentication returned HTTP {0}")]
    AuthStatus(u16),
    #[error("CyberArk Conjur returned HTTP {0}")]
    Status(u16),
    #[error(
        "CyberArk credentials are missing: set CYBERARK_API_KEY or both CYBERARK_CLIENT_CERT and CYBERARK_CLIENT_KEY"
    )]
    MissingCredentials,
    #[error("CyberArk client certificate could not be loaded")]
    ClientCertificate(
        #[source]
        #[redact]
        litellm_http::Error,
    ),
    #[error("CyberArk Conjur HTTP client could not be built")]
    Client(
        #[source]
        #[redact]
        litellm_http::Error,
    ),
    #[error("invalid refresh interval")]
    RefreshInterval(#[source] std::num::ParseIntError),
    #[error("invalid CyberArk Conjur endpoint")]
    Endpoint(#[source] url::ParseError),
    #[error("CyberArk secret manager requires an enterprise license")]
    EnterpriseRequired,
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
}

impl Error {
    pub fn is_unsafe_secret_name(&self) -> bool {
        match self {
            Self::Operation(error) => error.is_unsafe_secret_name(),
            Self::Timeout
            | Self::Http(_)
            | Self::AuthStatus(_)
            | Self::Status(_)
            | Self::MissingCredentials
            | Self::ClientCertificate(_)
            | Self::Client(_)
            | Self::RefreshInterval(_)
            | Self::Endpoint(_)
            | Self::EnterpriseRequired => false,
        }
    }
}

#[derive(thiserror::Error, veil::Redact)]
#[error("CyberArk Conjur write failed")]
pub struct WriteFailure {
    pub source: Error,
    #[redact]
    pub request_url: Option<reqwest::Url>,
    pub authentication: bool,
}

impl WriteFailure {
    pub(crate) fn local(source: Error) -> Self {
        Self {
            source,
            request_url: None,
            authentication: false,
        }
    }

    pub(crate) fn request(source: Error, url: reqwest::Url) -> Self {
        Self {
            source,
            request_url: Some(url),
            authentication: false,
        }
    }
}
