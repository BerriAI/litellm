#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid request: {0}")]
    InvalidRequest(#[source] ErrorDetail),
    #[error("invalid response: {0}")]
    InvalidResponse(#[source] ErrorDetail),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ErrorDetail {
    #[error("{0}")]
    Message(String),
    #[error("invalid {subject}: {source}")]
    Invalid {
        subject: &'static str,
        #[source]
        source: ErrorSource,
    },
    #[error("{operation} failed: {source}")]
    Failed {
        operation: &'static str,
        #[source]
        source: ErrorSource,
    },
    #[error("{field} must be {expected}, got {actual}")]
    InvalidValue {
        field: &'static str,
        expected: &'static str,
        actual: serde_json::Value,
    },
    #[error("Unmapped {field}: {actual}. Must be one of: {}.", .choices.iter().map(|choice| format!("'{choice}'")).collect::<Vec<_>>().join(", "))]
    InvalidChoice {
        field: &'static str,
        actual: String,
        choices: Vec<&'static str>,
    },
    #[error("{field}='{value}' is not supported by this model. Got model: {model}")]
    UnsupportedValue {
        field: &'static str,
        value: &'static str,
        model: String,
    },
    #[error(
        "{model} does not support {param}={value}. {hint}To drop unsupported params, set `litellm.drop_params = True`."
    )]
    UnsupportedParameter {
        model: String,
        param: String,
        value: String,
        hint: String,
    },
    #[error("invalid {subject} on line {line}: {source}")]
    InvalidLine {
        subject: &'static str,
        line: usize,
        #[source]
        source: ErrorSource,
    },
    #[error("{operation} failed: {detail}")]
    RemoteFailure {
        operation: &'static str,
        detail: serde_json::Value,
    },
}

impl ErrorDetail {
    pub fn invalid(
        subject: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Invalid {
            subject,
            source: ErrorSource::new(source),
        }
    }

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
