//! Retry decisions: how many retries a `RetryPolicy` grants for a failure, and whether a failure
//! is retried in the same group at all. Mirrors `get_num_retries_from_retry_policy` and
//! `resolve_retry_policy` in `litellm_core_utils/get_retry_from_policy.py`, and
//! `Router.should_retry_this_error`.

use std::collections::BTreeMap;

use crate::{
    config::RetryPolicy,
    error::{AttemptError, FailureKind},
    health::should_retry_status,
};

/// `resolve_retry_policy`: the group's own policy when `model_group_retry_policy` has one, else
/// the router's `retry_policy`.
pub fn resolve_retry_policy<'a>(
    retry_policy: Option<&'a RetryPolicy>,
    model_group: &str,
    model_group_retry_policy: Option<&'a BTreeMap<String, RetryPolicy>>,
) -> Option<&'a RetryPolicy> {
    model_group_retry_policy
        .and_then(|policies| policies.get(model_group))
        .or(retry_policy)
}

impl RetryPolicy {
    /// `get_num_retries_from_retry_policy`: `DefaultRetries` for an `UnsupportedEndpointError`;
    /// else `NotFoundErrorRetries` for any 404 answer, then the most specific exception class
    /// the policy names, then `DefaultRetries`.
    pub fn retries_for(&self, failure: &AttemptError) -> Option<u32> {
        if failure.is_a("UnsupportedEndpointError") {
            return self.DefaultRetries;
        }
        let not_found = (failure.status == Some(404))
            .then_some(self.NotFoundErrorRetries)
            .flatten();
        not_found
            .or_else(|| {
                failure
                    .class_names()
                    .iter()
                    .find_map(|class| self.field_for(class))
            })
            .or(self.DefaultRetries)
    }

    /// `_RETRIES_BY_EXCEPTION_TYPE`: the policy field an exception class reads.
    fn field_for(&self, class: &str) -> Option<u32> {
        match class {
            "AuthenticationError" => self.AuthenticationErrorRetries,
            "Timeout" => self.TimeoutErrorRetries,
            "RateLimitError" => self.RateLimitErrorRetries,
            "ContentPolicyViolationError" => self.ContentPolicyViolationErrorRetries,
            "BadRequestError" => self.BadRequestErrorRetries,
            "ServiceUnavailableError" => self.ServiceUnavailableErrorRetries,
            "InternalServerError" => self.InternalServerErrorRetries,
            _ => None,
        }
    }
}

/// What `should_retry_this_error` knows besides the failure.
#[derive(Clone, Copy, Debug, Default)]
pub struct RetryContext {
    pub healthy: usize,
    pub all: usize,
    /// `context_window_fallbacks is not None`.
    pub has_context_window_fallbacks: bool,
    /// `content_policy_fallbacks is not None`.
    pub has_content_policy_fallbacks: bool,
    /// `regular_fallbacks is not None and len(regular_fallbacks) > 0`.
    pub has_regular_fallbacks: bool,
}

/// `should_retry_this_error`, as a predicate: `false` where Python raises the error.
pub fn should_retry(failure: &AttemptError, ctx: RetryContext) -> bool {
    if failure.is_a("ContextWindowExceededError") && ctx.has_context_window_fallbacks {
        return false;
    }
    if failure.is_a("ContentPolicyViolationError") && ctx.has_content_policy_fallbacks {
        return false;
    }
    if let Some(status) = failure.status
        && !should_retry_status(status)
        && !matches!(status, 401 | 403)
        && !failure.is_a("UnsupportedEndpointError")
    {
        return false;
    }
    if failure.is_a("NotFoundError") {
        return false;
    }
    if failure.is_a("RateLimitError") && ctx.healthy == 0 && ctx.has_regular_fallbacks {
        return false;
    }
    if (failure.is_a("AuthenticationError") || failure.is_a("PermissionDeniedError"))
        && ctx.all <= 1
    {
        return false;
    }
    ctx.healthy > 0
}

/// `ContentPolicyViolationError`'s classes, most specific first.
pub(crate) fn content_policy_classes() -> Vec<std::sync::Arc<str>> {
    [
        "ContentPolicyViolationError",
        "BadRequestError",
        "APIStatusError",
        "APIError",
        "OpenAIError",
        "Exception",
        "BaseException",
        "object",
    ]
    .into_iter()
    .map(Into::into)
    .collect()
}

/// The exception classes a failure kind stands for, most specific first, for a host that
/// reports no class names (`litellm/exceptions.py`).
pub(crate) fn kind_classes(kind: FailureKind) -> &'static [&'static str] {
    match kind {
        FailureKind::RateLimit => &["RateLimitError"],
        FailureKind::ContextWindowExceeded => &["ContextWindowExceededError", "BadRequestError"],
        FailureKind::ContentPolicyViolation => &["ContentPolicyViolationError", "BadRequestError"],
        FailureKind::Authentication => &["AuthenticationError"],
        FailureKind::PermissionDenied => &["PermissionDeniedError"],
        FailureKind::NotFound => &["NotFoundError"],
        FailureKind::BadRequest => &["BadRequestError"],
        FailureKind::Timeout => &["Timeout", "APITimeoutError", "APIConnectionError"],
        FailureKind::ApiConnection => &["APIConnectionError"],
        FailureKind::InternalServer => &["InternalServerError"],
        FailureKind::ServiceUnavailable => &["ServiceUnavailableError"],
        FailureKind::BadGateway => &["BadGatewayError"],
        FailureKind::GuardrailIntervention
        | FailureKind::Declined
        | FailureKind::CallerCredential
        | FailureKind::Other => &[],
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn failure(kind: FailureKind, status: Option<u16>, classes: &[&str]) -> AttemptError {
        let mut failure = AttemptError::new(kind, status);
        failure.classes = classes.iter().map(|class| (*class).into()).collect();
        failure
    }

    #[rstest]
    #[case::own_field(failure(FailureKind::InternalServer, Some(500), &[]), Some(4))]
    #[case::default(failure(FailureKind::BadGateway, Some(502), &[]), Some(7))]
    #[case::not_found_status(failure(FailureKind::Other, Some(404), &["APIError"]), Some(1))]
    #[case::unsupported_endpoint(
        failure(FailureKind::Declined, Some(400), &["UnsupportedEndpointError", "BadRequestError"]),
        Some(7)
    )]
    #[case::malformed_reads_bad_request(
        failure(FailureKind::Declined, Some(400), &["MalformedRequestFieldError", "BadRequestError"]),
        Some(3)
    )]
    fn retries_follow_the_most_specific_class(
        #[case] failure: AttemptError,
        #[case] expected: Option<u32>,
    ) {
        let policy = RetryPolicy {
            InternalServerErrorRetries: Some(4),
            BadRequestErrorRetries: Some(3),
            NotFoundErrorRetries: Some(1),
            DefaultRetries: Some(7),
            ..RetryPolicy::default()
        };
        assert_eq!(policy.retries_for(&failure), expected);
    }

    #[rstest]
    #[case::rate_limit_with_healthy(FailureKind::RateLimit, Some(429), 1, 2, false, true)]
    #[case::rate_limit_none_healthy_with_fallbacks(
        FailureKind::RateLimit,
        Some(429),
        0,
        2,
        true,
        false
    )]
    #[case::bad_request(FailureKind::BadRequest, Some(400), 2, 2, false, false)]
    #[case::auth_lone_deployment(FailureKind::Authentication, Some(401), 1, 1, false, false)]
    #[case::auth_with_siblings(FailureKind::Authentication, Some(401), 1, 2, false, true)]
    #[case::not_found(FailureKind::NotFound, Some(404), 2, 2, false, false)]
    #[case::server_error_no_healthy(FailureKind::InternalServer, Some(500), 0, 2, false, false)]
    fn should_retry_follows_python(
        #[case] kind: FailureKind,
        #[case] status: Option<u16>,
        #[case] healthy: usize,
        #[case] all: usize,
        #[case] has_regular_fallbacks: bool,
        #[case] expected: bool,
    ) {
        let ctx = RetryContext {
            healthy,
            all,
            has_regular_fallbacks,
            ..RetryContext::default()
        };
        assert_eq!(
            should_retry(&AttemptError::new(kind, status), ctx),
            expected
        );
    }
}
