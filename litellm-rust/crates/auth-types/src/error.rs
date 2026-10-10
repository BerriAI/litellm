use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("invalid authentication configuration: {0}")]
    InvalidConfiguration(#[source] ErrorDetail),
    #[error("credential acquisition failed: {0}")]
    CredentialAcquisition(#[source] ErrorDetail),
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
    #[error("Missing {provider} {} - {}", .spec.setting, .spec.guidance())]
    MissingParam {
        provider: &'static str,
        spec: &'static crate::ParamSpec,
    },
    #[error("Missing {provider} API Base - {guidance}")]
    MissingApiBase {
        provider: &'static str,
        guidance: &'static str,
    },
    #[error("invalid authentication header")]
    InvalidHeader,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ErrorDetail {
    #[error("{0}")]
    Message(String),
    #[error("{field} must be {expected}")]
    InvalidType {
        field: String,
        expected: &'static str,
    },
    #[error("{subject} cannot be empty")]
    Empty { subject: String },
    #[error("credential header {0} already exists")]
    DuplicateHeader(&'static str),
    #[error("{operation} failed: {source}")]
    Failed {
        operation: &'static str,
        #[source]
        source: ErrorSource,
    },
}

impl ErrorDetail {
    pub fn failed(
        operation: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Failed {
            operation,
            source: ErrorSource::new(source),
        }
    }
}

impl From<String> for ErrorDetail {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

impl From<&str> for ErrorDetail {
    fn from(message: &str) -> Self {
        Self::Message(message.into())
    }
}

#[derive(Clone, Debug)]
pub struct ErrorSource(std::sync::Arc<dyn std::error::Error + Send + Sync>);

impl std::ops::Deref for ErrorSource {
    type Target = dyn std::error::Error + Send + Sync;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl std::fmt::Display for ErrorSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, formatter)
    }
}

impl ErrorSource {
    pub fn new(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self(std::sync::Arc::new(error))
    }
}

impl PartialEq for ErrorSource {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ErrorSource {}
