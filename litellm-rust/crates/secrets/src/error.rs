#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configured secret manager did not return a secret")]
    ManagedSecretMissing,
    #[error("native secret backend is unavailable for this system")]
    NativeBackendUnavailable,
    #[error("encrypted environment value is missing")]
    MissingCiphertext,
    #[error("ciphertext is not valid base64 for the configured manager")]
    InvalidCiphertext,
    #[error("decrypted value is not UTF-8")]
    Utf8,
    #[error("unsupported OIDC provider or missing build feature")]
    UnsupportedOidc,
    #[error("OIDC reference requires a provider and audience")]
    InvalidOidc,
    #[error("OIDC environment variable is missing")]
    MissingEnvironment,
    #[error("OIDC request failed")]
    OidcHttp,
    #[error("OIDC provider returned HTTP {0}")]
    OidcStatus(u16),
    #[error("OIDC response is invalid")]
    OidcResponse,
    #[error("OIDC file path must be absolute and within the credential allowlist")]
    UnsafeOidcPath,
    #[error("OIDC file could not be read")]
    OidcFile,
    #[error("secret cannot be converted to {expected}")]
    TypeMismatch { expected: &'static str },
    #[error("external secret manager failed")]
    ExternalManager(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("external secret manager read failed")]
    ExternalRead(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[cfg(feature = "aws")]
    #[error(transparent)]
    Aws(#[from] litellm_secrets_aws::Error),
    #[cfg(feature = "google")]
    #[error(transparent)]
    Google(#[from] litellm_secrets_google::Error),
    #[cfg(feature = "hashicorp")]
    #[error(transparent)]
    Hashicorp(#[from] litellm_secrets_hashicorp::Error),
    #[cfg(feature = "azure")]
    #[error(transparent)]
    Azure(#[from] litellm_secrets_azure::Error),
    #[cfg(feature = "cyberark")]
    #[error(transparent)]
    Cyberark(#[from] litellm_secrets_cyberark::Error),
}
