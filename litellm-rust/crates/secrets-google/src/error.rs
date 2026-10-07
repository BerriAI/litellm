#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
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
        reqwest::Error,
    ),
    #[error("Google Secret Manager returned HTTP {0}")]
    Status(u16),
    #[error("Google Secret Manager returned no payload")]
    MissingPayload,
    #[error("required environment variable is missing: {0}")]
    MissingEnvironment(&'static str),
    #[error("invalid refresh interval")]
    RefreshInterval,
    #[error("payload is not valid base64")]
    Base64(#[from] base64::DecodeError),
    #[error("decrypted value is not UTF-8")]
    Utf8,
    #[error("invalid Google Secret Manager endpoint")]
    Endpoint,
    #[error("Google Secret Manager requires an enterprise license")]
    EnterpriseRequired,
}
