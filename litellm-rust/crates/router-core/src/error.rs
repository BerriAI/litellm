//! The router's errors: [`ConfigError`] when a router is built or changed, [`Error`] when a call
//! or a selection fails, and [`AttemptError`], the typed facts of one failed attempt.

use std::{any::Any, fmt, sync::Arc, time::Duration};

use crate::config::{Op, RoutingStrategy};

/// What a host cannot serve, found when the router is built rather than partway through a call.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Unsupported {
    #[error("routing strategy `{0}` needs a host that can select")]
    Strategy(RoutingStrategy),
    #[error("routing plugin `{0}` needs a host that can select")]
    Plugin(String),
    #[error("operation `{op}` on provider `{provider}` has no executor")]
    Executor { op: Op, provider: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("the router config asks for something this host cannot serve")]
    Unsupported(#[from] Unsupported),
    #[error("deployment `{model_name}` has invalid litellm_params")]
    InvalidDeployment {
        model_name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("deployment id `{id}` is configured more than once")]
    DuplicateId { id: String },
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the router config is invalid")]
    Config(#[from] ConfigError),
    #[error("no deployment serves model `{model}`")]
    UnknownModel { model: String },
    #[error("no deployment of `{model}` is available")]
    NoHealthyDeployments { model: String },
    #[error("the attempt failed")]
    Attempt(#[from] AttemptError),
    #[error("the routing cache failed")]
    Cache(#[from] litellm_cache::Error),
    #[error("the router was retired")]
    Retired,
}

impl Error {
    /// The failed attempt behind this error, when a deployment was called.
    pub fn attempt(&self) -> Option<&AttemptError> {
        match self {
            Self::Attempt(attempt) => Some(attempt),
            Self::Config(_)
            | Self::UnknownModel { .. }
            | Self::NoHealthyDeployments { .. }
            | Self::Cache(_)
            | Self::Retired => None,
        }
    }
}

/// The class of a failed attempt, as litellm's exception mapping names it.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum FailureKind {
    #[strum(serialize = "RateLimitError")]
    RateLimit,
    #[strum(serialize = "ContextWindowExceededError")]
    ContextWindowExceeded,
    #[strum(serialize = "ContentPolicyViolationError")]
    ContentPolicyViolation,
    #[strum(serialize = "AuthenticationError")]
    Authentication,
    #[strum(serialize = "PermissionDeniedError")]
    PermissionDenied,
    #[strum(serialize = "NotFoundError")]
    NotFound,
    #[strum(serialize = "BadRequestError")]
    BadRequest,
    #[strum(serialize = "Timeout")]
    Timeout,
    #[strum(serialize = "APIConnectionError")]
    ApiConnection,
    #[strum(serialize = "InternalServerError")]
    InternalServer,
    #[strum(serialize = "ServiceUnavailableError")]
    ServiceUnavailable,
    #[strum(serialize = "BadGatewayError")]
    BadGateway,
    /// A guardrail blocked the request: never retried, never a fallback.
    #[strum(serialize = "GuardrailIntervention")]
    GuardrailIntervention,
    /// litellm rejected the request before calling the deployment (`MalformedRequestFieldError`,
    /// `UnsupportedEndpointError`): never cools the deployment down.
    #[strum(serialize = "Declined")]
    Declined,
    /// The caller's own credential failed, not the deployment's.
    #[strum(serialize = "CallerCredentialError")]
    CallerCredential,
    #[strum(serialize = "APIError")]
    Other,
}

/// An opaque value the host hands through the router unchanged, such as the original Python
/// exception or response object.
#[derive(Clone)]
pub struct Payload(pub Arc<dyn Any + Send + Sync>);

impl fmt::Debug for Payload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Payload(..)")
    }
}

/// One failed attempt, as typed facts the retry, fallback and cooldown rules decide by.
#[derive(Clone, Debug, thiserror::Error)]
#[error("the deployment call failed with {kind}")]
pub struct AttemptError {
    pub kind: FailureKind,
    pub status: Option<u16>,
    /// The provider's `retry-after` (or `retry-after-ms`) header.
    pub retry_after: Option<Duration>,
    /// A deployment-level `num_retries` the failure carries (Python's `exception.num_retries`).
    pub num_retries: Option<u32>,
    /// The provider's error text, kept only for the cooldown record Python writes
    /// (`exception_received`, masked past 50 characters).
    pub detail: Option<Arc<str>>,
    pub payload: Option<Payload>,
}

impl AttemptError {
    pub fn new(kind: FailureKind, status: Option<u16>) -> Self {
        Self {
            kind,
            status,
            retry_after: None,
            num_retries: None,
            detail: None,
            payload: None,
        }
    }

    #[must_use]
    pub fn with_retry_after(self, retry_after: Duration) -> Self {
        Self {
            retry_after: Some(retry_after),
            ..self
        }
    }

    /// The status Python's cooldown rules read: the provider's, else 500.
    pub fn cooldown_status(&self) -> u16 {
        self.status.unwrap_or(500)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::rate_limit("RateLimitError", FailureKind::RateLimit)]
    #[case::context_window("ContextWindowExceededError", FailureKind::ContextWindowExceeded)]
    #[case::caller_credential("CallerCredentialError", FailureKind::CallerCredential)]
    #[case::other("APIError", FailureKind::Other)]
    fn failure_kind_round_trips_its_exception_name(#[case] wire: &str, #[case] kind: FailureKind) {
        assert_eq!(wire.parse::<FailureKind>().unwrap(), kind);
        assert_eq!(<&'static str>::from(kind), wire);
    }

    #[rstest]
    #[case::empty("")]
    #[case::lowercase("ratelimiterror")]
    fn failure_kind_rejects_unknown_names(#[case] wire: &str) {
        assert!(wire.parse::<FailureKind>().is_err());
    }
}
