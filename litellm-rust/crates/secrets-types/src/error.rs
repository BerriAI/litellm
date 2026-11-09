#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("secret name contains an unsafe path segment or control character")]
    UnsafeSecretName,
    #[error("current secret was not found")]
    CurrentSecretMissing,
    #[error("new secret could not be verified")]
    NewSecretMissing,
    #[error("new secret does not match the replacement")]
    NewSecretMismatch,
    #[error("secret manager received an incompatible operation context")]
    InvalidOperationContext,
}

impl Error {
    pub fn is_unsafe_secret_name(&self) -> bool {
        match self {
            Self::UnsafeSecretName => true,
            Self::CurrentSecretMissing
            | Self::NewSecretMissing
            | Self::NewSecretMismatch
            | Self::InvalidOperationContext => false,
        }
    }
}
