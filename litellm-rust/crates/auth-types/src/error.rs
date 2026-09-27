use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("invalid authentication configuration: {0}")]
    InvalidConfiguration(String),
    #[error("credential acquisition failed: {0}")]
    CredentialAcquisition(String),
    #[error("credential caller failed: {0}")]
    EmptyCallerCredential(&'static str),
    #[error("{0}")]
    ProviderAuthentication(String),
    #[error("credential acquisition failed: {}", .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    CredentialChain(Vec<Error>),
    #[error(
        "Missing {provider} API Key - Set `api_key` or the {environment_variable} environment variable"
    )]
    MissingApiKey {
        provider: &'static str,
        environment_variable: &'static str,
    },
    #[error("Missing {provider} API Base - {guidance}")]
    MissingApiBase {
        provider: &'static str,
        guidance: &'static str,
    },
    #[error("invalid authentication header")]
    InvalidHeader,
}
