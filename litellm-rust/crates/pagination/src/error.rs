use serde::{Deserialize, Serialize};

/// Stable, machine readable failure codes shared by every paginated read and every adapter.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    InvalidRequest,
    InvalidCursor,
    TraversalExpired,
    TraversalChanged,
    ResourceTooLarge,
    BudgetExceeded,
    ViewNotReady,
    Unavailable,
    Busy,
}

impl FailureCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::InvalidCursor => "invalid_cursor",
            Self::TraversalExpired => "traversal_expired",
            Self::TraversalChanged => "traversal_changed",
            Self::ResourceTooLarge => "resource_too_large",
            Self::BudgetExceeded => "budget_exceeded",
            Self::ViewNotReady => "view_not_ready",
            Self::Unavailable => "unavailable",
            Self::Busy => "busy",
        }
    }

    /// Whether repeating the same request unchanged can succeed later.
    pub const fn is_transient(self) -> bool {
        matches!(self, Self::ViewNotReady | Self::Unavailable | Self::Busy)
    }

    /// Whether the client must discard loaded pages and restart from the first page.
    pub const fn restarts_traversal(self) -> bool {
        matches!(self, Self::TraversalExpired | Self::TraversalChanged)
    }
}

/// A public read failure: the stable code plus a message safe to show to the caller.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Failure {
    pub code: FailureCode,
    pub message: String,
}

impl Failure {
    pub fn new(code: FailureCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid or unrecognized cursor; restart from the first page")]
    InvalidCursor,
    #[error("This page of results has expired; restart from the first page")]
    TraversalExpired,
    #[error("Results changed while paging; refresh to continue")]
    TraversalChanged,
    #[error("One item exceeds the response size limit")]
    ResourceTooLarge,
    #[error("Cursor signing keys must be nonempty")]
    InvalidKeys,
    #[error("Cursor encoding failed")]
    Encode(#[from] serde_json::Error),
}

impl Error {
    pub const fn code(&self) -> FailureCode {
        match self {
            Self::InvalidCursor => FailureCode::InvalidCursor,
            Self::TraversalExpired => FailureCode::TraversalExpired,
            Self::TraversalChanged => FailureCode::TraversalChanged,
            Self::ResourceTooLarge => FailureCode::ResourceTooLarge,
            Self::InvalidKeys | Self::Encode(_) => FailureCode::Unavailable,
        }
    }
}

impl From<&Error> for Failure {
    fn from(error: &Error) -> Self {
        Self::new(error.code(), error.to_string())
    }
}
