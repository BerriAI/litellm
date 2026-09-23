#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error("HashiCorp Vault requires an enterprise license")]
    EnterpriseRequired,
    #[error("invalid secret name")]
    InvalidSecretName(litellm_secrets_types::Error),
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
    #[error("HashiCorp Vault client failed")]
    Client(
        #[from]
        #[redact]
        vaultrs::error::ClientError,
    ),
    #[error("HashiCorp Vault client settings are invalid: {message}")]
    ClientSettings { message: String },
    #[error("HashiCorp Vault TLS identity could not be configured for {path}: {message}")]
    TlsIdentity {
        path: std::path::PathBuf,
        message: String,
    },
    #[error("HashiCorp Vault login returned HTTP {status}")]
    LoginStatus { status: u16 },
    #[error("HashiCorp Vault login response is malformed")]
    MalformedLogin,
    #[error("HashiCorp Vault authentication is not configured")]
    NoAuthConfigured,
    #[error("HashiCorp Vault returned HTTP {status}")]
    Status { status: u16 },
    #[error("HashiCorp Vault response payload is malformed")]
    MalformedPayload,
    #[error("HashiCorp Vault secret value is not a string")]
    NonStringValue,
    #[error("HashiCorp Vault data key conflicts with description")]
    DataKeyConflictsWithDescription,
    #[error("HashiCorp Vault secret version exceeds CAS range")]
    CasVersionOverflow,
    #[error("HashiCorp Vault operation timed out")]
    Timeout,
    #[error("invalid HashiCorp Vault refresh interval")]
    RefreshInterval,
}
