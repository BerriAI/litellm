use thiserror::Error as ThisError;

#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
pub enum Error {
    #[error("AWS profile credentials failed: {0}")]
    ProfileCredentials(String),
    #[error("AWS default credentials failed: {0}")]
    DefaultCredentials(String),
    #[error("AWS role credentials failed: {0}")]
    RoleCredentials(String),
    #[error("AWS web identity credentials failed: {0}")]
    WebIdentityCredentials(String),
    #[error("AWS web identity response had no credentials")]
    MissingWebIdentityCredentials,
    #[error("AWS web identity expiration was invalid: {0}")]
    InvalidWebIdentityExpiration(String),
    #[error("AWS signing parameters failed: {0}")]
    SigningParameters(String),
    #[error("AWS signable request failed: {0}")]
    SignableRequest(String),
    #[error("AWS request signing failed: {0}")]
    RequestSigning(String),
}
