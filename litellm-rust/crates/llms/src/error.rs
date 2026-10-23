use litellm_error::Source;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid request")]
    InvalidRequest(#[source] ErrorDetail),
    #[error("invalid response")]
    InvalidResponse(#[source] ErrorDetail),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ErrorDetail {
    #[error("invalid {subject}")]
    Invalid {
        subject: &'static str,
        #[source]
        cause: Source,
    },
    #[error("{operation} failed")]
    Failed {
        operation: &'static str,
        #[source]
        cause: Source,
    },
    #[error("{subject} must be {expected}")]
    Expected {
        subject: &'static str,
        expected: &'static str,
    },
    #[error("{subject} is missing {field}")]
    MissingField {
        subject: &'static str,
        field: &'static str,
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
    #[error(
        "Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({{LLM_PROVIDER}}_API_KEY) or via params"
    )]
    MissingApiBase,
    #[error("the host call raised an exception")]
    HostCall,
    #[error("invalid {subject} on line {line}")]
    InvalidLine {
        subject: &'static str,
        line: usize,
        #[source]
        cause: Source,
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
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Invalid {
            subject,
            cause: Source::new(cause),
        }
    }

    pub fn failed(
        operation: &'static str,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Failed {
            operation,
            cause: Source::new(cause),
        }
    }

    pub fn invalid_line(
        subject: &'static str,
        line: usize,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::InvalidLine {
            subject,
            line,
            cause: Source::new(cause),
        }
    }
}
