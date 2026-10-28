#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
    #[error("secret manager operation timed out")]
    Timeout,
    #[error("Google KMS client configuration failed")]
    Client(
        #[from]
        #[redact]
        google_cloud_gax::client_builder::Error,
    ),
    #[error("Google authentication failed")]
    Auth(
        #[from]
        #[redact]
        litellm_auth_types::Error,
    ),
    #[error("Google KMS request failed")]
    Kms(
        #[from]
        #[redact]
        google_cloud_gax::error::Error,
    ),
    #[error("Google Secret Manager HTTP request failed")]
    Http(
        #[from]
        #[redact]
        litellm_http::Error,
    ),
    #[error("Google Secret Manager returned HTTP {0}")]
    Status(u16),
    #[error("Google Secret Manager returned no payload")]
    MissingPayload,
    #[error("required environment variable is missing: {0}")]
    MissingEnvironment(&'static str),
    #[error("invalid refresh interval")]
    RefreshInterval(#[source] std::num::ParseIntError),
    #[error("payload is not valid base64")]
    Base64(
        #[from]
        #[redact]
        base64::DecodeError,
    ),
    #[error("Google Secret Manager payload checksum mismatch")]
    Checksum(#[source] Option<std::num::ParseIntError>),
    #[error("decrypted value is not UTF-8")]
    Utf8(#[source] std::str::Utf8Error),
    #[error("invalid Google Secret Manager endpoint")]
    Endpoint(#[source] url::ParseError),
    #[error("Google Secret Manager requires an enterprise license")]
    EnterpriseRequired,
}

impl Error {
    pub fn is_not_found(&self) -> bool {
        match self {
            Self::Status(status) => *status == 404,
            Self::Operation(_)
            | Self::Timeout
            | Self::Client(_)
            | Self::Auth(_)
            | Self::Kms(_)
            | Self::Http(_)
            | Self::MissingPayload
            | Self::MissingEnvironment(_)
            | Self::RefreshInterval(_)
            | Self::Base64(_)
            | Self::Checksum(_)
            | Self::Utf8(_)
            | Self::Endpoint(_)
            | Self::EnterpriseRequired => false,
        }
    }
}
