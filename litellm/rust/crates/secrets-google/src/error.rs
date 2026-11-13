use litellm_secret_value::Redacted;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
    #[error("secret manager operation timed out")]
    Timeout,
    #[error("Google KMS client configuration failed")]
    Client(#[source] Redacted<google_cloud_gax::client_builder::Error>),
    #[error("Google authentication failed")]
    Auth(#[source] Redacted<litellm_auth_types::Error>),
    #[error("Google KMS request failed")]
    Kms(#[source] Redacted<google_cloud_gax::error::Error>),
    #[error("Google Secret Manager HTTP request failed")]
    Http(#[source] Redacted<litellm_http::Error>),
    #[error("Google Secret Manager returned HTTP {0}")]
    Status(u16),
    #[error("Google Secret Manager returned no payload")]
    MissingPayload,
    #[error("required environment variable is missing: {0}")]
    MissingEnvironment(&'static str),
    #[error("invalid refresh interval")]
    RefreshInterval(#[source] std::num::ParseIntError),
    #[error("payload is not valid base64")]
    Base64(#[source] Redacted<base64::DecodeError>),
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

impl From<google_cloud_gax::client_builder::Error> for Error {
    fn from(error: google_cloud_gax::client_builder::Error) -> Self {
        Self::Client(Redacted::new(error))
    }
}

impl From<litellm_auth_types::Error> for Error {
    fn from(error: litellm_auth_types::Error) -> Self {
        Self::Auth(Redacted::new(error))
    }
}

impl From<google_cloud_gax::error::Error> for Error {
    fn from(error: google_cloud_gax::error::Error) -> Self {
        Self::Kms(Redacted::new(error))
    }
}

impl From<litellm_http::Error> for Error {
    fn from(error: litellm_http::Error) -> Self {
        Self::Http(Redacted::new(error))
    }
}

impl From<base64::DecodeError> for Error {
    fn from(error: base64::DecodeError) -> Self {
        Self::Base64(Redacted::new(error))
    }
}
