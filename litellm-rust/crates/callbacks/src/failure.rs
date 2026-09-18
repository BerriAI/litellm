use std::fmt;
use std::time::Duration;

/// What kind of failure an attempt reported: a fact about the error, owned by the route
/// that produced it. The loop that runs attempts decides what to do about it; a host
/// records it (cooldowns, metrics) before that loop continues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailureClass {
    /// 429.
    RateLimited,
    /// 408, or the request timed out before a response arrived.
    Timeout,
    /// 409.
    Conflict,
    /// 503.
    ServiceUnavailable,
    /// Any other 5xx.
    InternalServer,
    /// The provider could not be reached, so there is no status.
    Connection,
    /// 401 or 403.
    Authentication,
    /// 404.
    NotFound,
    /// The prompt does not fit the model's context window.
    ContextWindow,
    /// The provider's content filter rejected the request or the response.
    ContentPolicy,
    /// Any other 4xx.
    BadRequest,
}

impl FailureClass {
    /// The class an HTTP status alone implies. Routes refine it where the body says more:
    /// context-window and content-policy errors are 400s with a recognisable message.
    pub const fn from_status(status: u16) -> Self {
        match status {
            408 => Self::Timeout,
            409 => Self::Conflict,
            429 => Self::RateLimited,
            401 | 403 => Self::Authentication,
            404 => Self::NotFound,
            503 => Self::ServiceUnavailable,
            400..=499 => Self::BadRequest,
            _ => Self::InternalServer,
        }
    }

    /// Whether the status alone makes the failure worth retrying: 408, 409, 429 and every
    /// 5xx, as Python's `litellm._should_retry` has it. A connection failure has no
    /// status and is retried as well.
    pub const fn retryable_status(self) -> bool {
        matches!(
            self,
            Self::RateLimited
                | Self::Timeout
                | Self::Conflict
                | Self::ServiceUnavailable
                | Self::InternalServer
                | Self::Connection
        )
    }

    /// The class a per-class retry policy falls back to, the way Python walks the
    /// exception's MRO: context-window and content-policy errors are bad requests.
    pub const fn parent(self) -> Option<Self> {
        match self {
            Self::ContextWindow | Self::ContentPolicy => Some(Self::BadRequest),
            _ => None,
        }
    }
}

impl fmt::Display for FailureClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
    }
}

/// A route error that knows what kind of failure it is. Implemented by each route's error
/// type, so any machine over that route can run as one attempt of a larger call.
pub trait Classified {
    fn class(&self) -> FailureClass;

    /// A provider `Retry-After`, when the response carried one.
    fn retry_after(&self) -> Option<Duration> {
        None
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(408, FailureClass::Timeout, true)]
    #[case(409, FailureClass::Conflict, true)]
    #[case(429, FailureClass::RateLimited, true)]
    #[case(500, FailureClass::InternalServer, true)]
    #[case(502, FailureClass::InternalServer, true)]
    #[case(503, FailureClass::ServiceUnavailable, true)]
    #[case(400, FailureClass::BadRequest, false)]
    #[case(401, FailureClass::Authentication, false)]
    #[case(403, FailureClass::Authentication, false)]
    #[case(404, FailureClass::NotFound, false)]
    #[case(422, FailureClass::BadRequest, false)]
    fn status_classes_follow_python_should_retry(
        #[case] status: u16,
        #[case] class: FailureClass,
        #[case] retryable: bool,
    ) {
        assert_eq!(FailureClass::from_status(status), class);
        assert_eq!(class.retryable_status(), retryable);
    }

    #[test]
    fn body_refined_classes_are_bad_requests_to_a_retry_policy() {
        assert_eq!(
            FailureClass::ContextWindow.parent(),
            Some(FailureClass::BadRequest)
        );
        assert_eq!(
            FailureClass::ContentPolicy.parent(),
            Some(FailureClass::BadRequest)
        );
        assert_eq!(FailureClass::BadRequest.parent(), None);
        assert!(!FailureClass::ContextWindow.retryable_status());
        assert!(FailureClass::Connection.retryable_status());
    }
}
