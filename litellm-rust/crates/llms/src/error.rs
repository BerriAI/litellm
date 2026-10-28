use litellm_core_utils::decline::Decline;
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
    #[error(transparent)]
    Declined(Decline),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
}

impl Error {
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidType { .. } | Self::MissingField(_) | Self::InvalidRequest(_) => true,
            Self::Auth(error) => error.is_request(),
            Self::Http(error) => error.is_request(),
            Self::InvalidResponse(_) | Self::Declined(_) => false,
        }
    }

    pub fn is_response(&self) -> bool {
        match self {
            Self::InvalidResponse(_) => true,
            Self::Http(error) => error.is_response(),
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::Declined(_)
            | Self::Auth(_) => false,
        }
    }

    pub fn is_missing_credential(&self) -> bool {
        match self {
            Self::Auth(error) => error.is_missing_credential(),
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::InvalidResponse(_)
            | Self::Declined(_)
            | Self::Http(_) => false,
        }
    }

    /// A transformation runs on both sides of the send, so only the transport and credential
    /// resolution, which always runs before it, can tell whether the request reached the provider.
    pub fn dispatched(&self) -> bool {
        match self {
            Self::Http(error) => error.dispatched(),
            Self::Auth(_) => false,
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::InvalidResponse(_) => true,
            Self::Declined(_) => false,
        }
    }

    /// Why the transformation rejected the caller's request, rendered as Python's own
    /// validation renders it, without this layer's "invalid request" prefix.
    pub fn rejected_request(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRequest(detail) => Some(detail),
            Self::InvalidType { .. } | Self::MissingField(_) => Some(self),
            Self::InvalidResponse(_) | Self::Declined(_) | Self::Auth(_) | Self::Http(_) => None,
        }
    }

    pub fn declined(&self) -> Option<&Decline> {
        match self {
            Self::Declined(decline) => Some(decline),
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::InvalidResponse(_)
            | Self::Auth(_)
            | Self::Http(_) => None,
        }
    }

    pub fn upstream_response(&self) -> Option<(u16, &str)> {
        match self {
            Self::Http(error) => error.upstream_response(),
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::InvalidResponse(_)
            | Self::Declined(_)
            | Self::Auth(_) => None,
        }
    }
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
    #[error("{route} requires at least one message")]
    NoMessages { route: &'static str },
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
    #[error("{0} is not handled by the rust path")]
    Unhandled(&'static str),
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
