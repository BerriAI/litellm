#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error("CyberArk Conjur operation timed out")]
    Timeout,
    #[error("CyberArk Conjur HTTP request failed")]
    Http(
        #[from]
        #[redact]
        reqwest::Error,
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
    ClientCertificate,
    #[error("invalid refresh interval")]
    RefreshInterval,
    #[error("invalid CyberArk Conjur endpoint")]
    Endpoint,
    #[error("CyberArk secret manager requires an enterprise license")]
    EnterpriseRequired,
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
}
