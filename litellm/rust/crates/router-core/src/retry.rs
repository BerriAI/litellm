//! Retry decisions: how many retries a `RetryPolicy` grants for a failure, and whether a failure
//! is retried in the same group at all. Mirrors `get_num_retries_from_retry_policy` and
//! `resolve_retry_policy` in `litellm_core_utils/get_retry_from_policy.py`, and
//! `Router.should_retry_this_error`.

use std::{collections::BTreeMap, time::Duration};

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
    pub untried: usize,
    pub all: usize,
    /// A `context_window_fallbacks` chain matches the failed group
    /// (`context_window_fallback_model_group`).
    pub has_context_window_fallbacks: bool,
    /// `content_policy_fallbacks is not None`.
    pub has_content_policy_fallbacks: bool,
    /// `regular_fallbacks is not None and len(regular_fallbacks) > 0`.
    pub has_regular_fallbacks: bool,
}

/// `should_retry_this_error`, as a predicate: `false` where Python raises the error.
pub fn should_retry(failure: &AttemptError, ctx: RetryContext) -> bool {
    if failure.is_a("ContextWindowExceededError") {
        return !ctx.has_context_window_fallbacks && ctx.untried > 0;
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

/// `INITIAL_RETRY_DELAY`, `MAX_RETRY_DELAY` and `JITTER` in `litellm/constants.py`.
const INITIAL_RETRY_DELAY: f64 = 0.5;
const MAX_RETRY_DELAY: f64 = 8.0;
const JITTER: f64 = 0.75;
/// `calculate_retry_after` obeys a provider's retry-after up to a minute.
const MAX_HONOURED_RETRY_AFTER: Duration = Duration::from_secs(60);

/// The sleep before a retry (`_time_to_sleep_before_retry` and `calculate_retry_after`), up to
/// its jitter: the loop draws that only when it sleeps, so a loop that stops first draws nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Backoff {
    /// Another healthy deployment takes the retry at once.
    Immediate,
    /// The provider's `retry-after`, before jitter.
    RetryAfter(Duration),
    /// The exponential backoff in seconds, before jitter.
    Exponential(f64),
}

impl Backoff {
    fn of(state: &RetryState<'_>) -> Self {
        if state.context.all != 1 && state.context.healthy > 0 {
            return Self::Immediate;
        }
        if let Some(retry_after) = state.failure.retry_after
            && !retry_after.is_zero()
            && retry_after <= MAX_HONOURED_RETRY_AFTER
        {
            return Self::RetryAfter(retry_after);
        }
        let doubled = INITIAL_RETRY_DELAY * 2f64.powi(state.attempted as i32);
        Self::Exponential(doubled.max(state.floor.as_secs_f64()).min(MAX_RETRY_DELAY))
    }

    /// The sleep itself. `draw` gives the jitter's share of `JITTER`, in `[0, 1)`.
    pub(crate) fn delay(self, draw: impl FnOnce() -> f64) -> Duration {
        match self {
            Self::Immediate => Duration::ZERO,
            Self::RetryAfter(retry_after) => retry_after + Duration::from_secs_f64(JITTER * draw()),
            Self::Exponential(seconds) => Duration::from_secs_f64(seconds + JITTER * draw()),
        }
    }
}

/// What a retry loop knows when it holds a failure: all [`retry_step`] decides from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RetryState<'a> {
    pub failure: &'a AttemptError,
    /// The retries the loop may make for this failure.
    pub budget: u32,
    /// The retries it already made.
    pub attempted: u32,
    /// A retry policy granted the budget, so `should_retry_this_error` is not asked.
    pub policy_applies: bool,
    pub context: RetryContext,
    /// The router's `retry_after`: the least an exponential backoff waits.
    pub floor: Duration,
}

/// What a retry loop does with the failure it holds. The loop keeps the effects: it sleeps,
/// attempts and reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RetryStep {
    /// Nothing retries the failure (`should_retry_this_error` raises it).
    Raise,
    /// The budget has a retry left: sleep the backoff, then attempt again.
    Retry { backoff: Backoff },
    /// Every retry the budget grants was made. The backoff is the sleep Python's unary loop still
    /// takes after its last failed retry. A stream's retries end without it.
    Exhausted { backoff: Backoff },
}

/// The one decision the unary retry loop and the stream retry loop share. How each resolves its
/// budget stays its own.
pub(crate) fn retry_step(state: RetryState<'_>) -> RetryStep {
    if !state.policy_applies && !should_retry(state.failure, state.context) {
        return RetryStep::Raise;
    }
    let backoff = Backoff::of(&state);
    if state.attempted >= state.budget {
        return RetryStep::Exhausted { backoff };
    }
    RetryStep::Retry { backoff }
}

/// `ContentPolicyViolationError`'s classes, most specific first.
pub(crate) fn content_policy_classes() -> Box<[std::sync::Arc<str>]> {
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
        FailureKind::CallerCredentialAuthentication => {
            &["CallerCredentialAuthenticationError", "AuthenticationError"]
        }
        FailureKind::CallerCredentialRateLimit => {
            &["CallerCredentialRateLimitError", "RateLimitError"]
        }
        FailureKind::GuardrailIntervention
        | FailureKind::Declined
        | FailureKind::QueueRejected
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

    fn state(failure: &AttemptError, budget: u32, attempted: u32) -> RetryState<'_> {
        RetryState {
            failure,
            budget,
            attempted,
            policy_applies: false,
            context: RetryContext {
                healthy: 1,
                all: 1,
                ..RetryContext::default()
            },
            floor: Duration::ZERO,
        }
    }

    fn server_error() -> AttemptError {
        AttemptError::new(FailureKind::InternalServer, Some(500))
    }

    fn bad_request() -> AttemptError {
        AttemptError::new(FailureKind::BadRequest, Some(400))
    }

    #[rstest]
    #[case::a_retry_is_left(server_error(), 2, 1, false, Some(false))]
    #[case::the_budget_is_spent(server_error(), 2, 2, false, Some(true))]
    #[case::the_budget_grants_none(server_error(), 0, 0, false, Some(true))]
    #[case::nothing_retries_the_failure(bad_request(), 2, 0, false, None)]
    #[case::a_spent_budget_still_raises_what_nothing_retries(bad_request(), 2, 2, false, None)]
    #[case::a_retry_policy_is_not_asked_whether_to_retry(bad_request(), 2, 0, true, Some(false))]
    #[case::a_retry_policy_still_runs_out(bad_request(), 2, 2, true, Some(true))]
    fn the_step_follows_the_failure_then_the_budget(
        #[case] failure: AttemptError,
        #[case] budget: u32,
        #[case] attempted: u32,
        #[case] policy_applies: bool,
        #[case] exhausted: Option<bool>,
    ) {
        let step = retry_step(RetryState {
            policy_applies,
            ..state(&failure, budget, attempted)
        });
        let expected = match exhausted {
            None => RetryStep::Raise,
            Some(false) => RetryStep::Retry {
                backoff: Backoff::Exponential(INITIAL_RETRY_DELAY * 2f64.powi(attempted as i32)),
            },
            Some(true) => RetryStep::Exhausted {
                backoff: Backoff::Exponential(INITIAL_RETRY_DELAY * 2f64.powi(attempted as i32)),
            },
        };
        assert_eq!(step, expected);
    }

    fn after(retry_after: Duration) -> AttemptError {
        let mut failure = server_error();
        failure.retry_after = Some(retry_after);
        failure
    }

    #[rstest]
    #[case::a_lone_deployment_starts_at_the_initial_delay(
        server_error(),
        1,
        1,
        0,
        0,
        Backoff::Exponential(0.5)
    )]
    #[case::each_retry_doubles_it(server_error(), 1, 1, 3, 0, Backoff::Exponential(4.0))]
    #[case::up_to_the_cap(server_error(), 1, 1, 9, 0, Backoff::Exponential(8.0))]
    #[case::the_routers_retry_after_is_the_floor(
        server_error(),
        1,
        1,
        0,
        3,
        Backoff::Exponential(3.0)
    )]
    #[case::the_floor_is_capped_too(server_error(), 1, 1, 0, 30, Backoff::Exponential(8.0))]
    #[case::another_healthy_deployment_retries_at_once(
        server_error(),
        1,
        2,
        3,
        3,
        Backoff::Immediate
    )]
    #[case::nothing_healthy_backs_off(server_error(), 0, 2, 1, 0, Backoff::Exponential(1.0))]
    #[case::a_group_that_lists_nothing_retries_at_once(
        server_error(),
        1,
        0,
        1,
        0,
        Backoff::Immediate
    )]
    #[case::the_providers_retry_after_is_obeyed(
        after(Duration::from_secs(5)),
        1,
        1,
        3,
        0,
        Backoff::RetryAfter(Duration::from_secs(5))
    )]
    #[case::up_to_a_minute(
        after(MAX_HONOURED_RETRY_AFTER),
        1,
        1,
        0,
        0,
        Backoff::RetryAfter(MAX_HONOURED_RETRY_AFTER)
    )]
    #[case::a_longer_one_is_ignored(
        after(Duration::from_secs(61)),
        1,
        1,
        1,
        0,
        Backoff::Exponential(1.0)
    )]
    #[case::a_zero_one_is_ignored(after(Duration::ZERO), 1, 1, 1, 0, Backoff::Exponential(1.0))]
    #[case::another_healthy_deployment_ignores_it(
        after(Duration::from_secs(5)),
        2,
        2,
        0,
        0,
        Backoff::Immediate
    )]
    fn the_backoff_follows_the_groups_health_then_the_failure(
        #[case] failure: AttemptError,
        #[case] healthy: usize,
        #[case] all: usize,
        #[case] attempted: u32,
        #[case] floor_seconds: u64,
        #[case] expected: Backoff,
    ) {
        let step = retry_step(RetryState {
            policy_applies: true,
            context: RetryContext {
                healthy,
                all,
                ..RetryContext::default()
            },
            floor: Duration::from_secs(floor_seconds),
            ..state(&failure, 10, attempted)
        });
        assert_eq!(step, RetryStep::Retry { backoff: expected });
    }

    #[rstest]
    #[case::the_retry_after_takes_jitter(
        Backoff::RetryAfter(Duration::from_secs(5)),
        Duration::from_millis(5_375)
    )]
    #[case::the_exponential_backoff_takes_jitter(
        Backoff::Exponential(2.0),
        Duration::from_millis(2_375)
    )]
    fn a_sleep_adds_its_share_of_the_jitter(#[case] backoff: Backoff, #[case] expected: Duration) {
        assert_eq!(backoff.delay(|| 0.5), expected);
    }

    #[test]
    fn an_immediate_retry_draws_no_jitter() {
        let delay = Backoff::Immediate.delay(|| panic!("an immediate retry drew jitter"));
        assert_eq!(delay, Duration::ZERO);
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
            untried: healthy,
            all,
            has_regular_fallbacks,
            ..RetryContext::default()
        };
        assert_eq!(
            should_retry(&AttemptError::new(kind, status), ctx),
            expected
        );
    }

    #[rstest]
    #[case::a_sibling_has_not_refused(2, 1, false, true)]
    #[case::every_healthy_deployment_refused(2, 0, false, false)]
    #[case::context_window_fallbacks_take_over(2, 1, true, false)]
    fn context_window_error_moves_to_a_deployment_that_has_not_refused(
        #[case] healthy: usize,
        #[case] untried: usize,
        #[case] has_context_window_fallbacks: bool,
        #[case] expected: bool,
    ) {
        let ctx = RetryContext {
            healthy,
            untried,
            all: 2,
            has_context_window_fallbacks,
            ..RetryContext::default()
        };
        let failure = AttemptError::new(FailureKind::ContextWindowExceeded, Some(400));
        assert_eq!(should_retry(&failure, ctx), expected);
    }
}
