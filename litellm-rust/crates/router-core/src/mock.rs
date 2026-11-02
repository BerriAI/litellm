//! The request's `mock_testing_*` flags: a failure the router raises itself, without calling a
//! deployment (`_handle_mock_testing_fallbacks` and `_handle_mock_testing_rate_limit_error`).

use serde::Deserialize;

use crate::error::{AttemptError, FailureKind};

/// Which mock failure a request asks for.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Deserialize, strum::IntoStaticStr, strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum MockFailure {
    /// `mock_testing_fallbacks`: an `InternalServerError` before the first level's retries.
    Fallbacks,
    /// `mock_testing_context_fallbacks`: a `ContextWindowExceededError`, likewise.
    ContextWindow,
    /// `mock_testing_content_policy_fallbacks`: a `ContentPolicyViolationError`, likewise.
    ContentPolicy,
    /// `mock_testing_rate_limit_error`: a `RateLimitError` in place of every level's first
    /// attempt, which the retry loop handles like a failed attempt.
    RateLimit,
}

impl MockFailure {
    /// Whether the failure replaces a level's walk (raised before its retries).
    pub fn before_retries(self) -> bool {
        !matches!(self, Self::RateLimit)
    }

    /// The failure as the rules see it.
    pub fn failure(self, num_retries: Option<u32>) -> AttemptError {
        let (kind, status) = match self {
            Self::Fallbacks => (FailureKind::InternalServer, 500),
            Self::ContextWindow => (FailureKind::ContextWindowExceeded, 400),
            Self::ContentPolicy => (FailureKind::ContentPolicyViolation, 400),
            Self::RateLimit => (FailureKind::RateLimit, 429),
        };
        let mut failure = AttemptError::new(kind, Some(status));
        failure.num_retries = num_retries;
        failure
    }
}
