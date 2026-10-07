//! The retry layer's decisions: `should_retry_this_error` and `_time_to_sleep_before_retry`.

use crate::{
    cooldown::status_is_retryable,
    failure::{Classified, ExceptionClass},
    random::PythonRandom,
    settings::{Fallbacks, Tunables},
};

pub struct RetryContext<'a> {
    pub healthy: usize,
    pub all: usize,
    pub context_window_fallbacks: Option<&'a Fallbacks>,
    pub content_policy_fallbacks: Option<&'a Fallbacks>,
    pub fallbacks: Option<&'a Fallbacks>,
}

/// `should_retry_this_error`, checked in Python's order.
pub fn should_retry(failure: &Classified, context: &RetryContext<'_>) -> bool {
    let no_retry = (failure.is(ExceptionClass::ContextWindowExceeded)
        && context.context_window_fallbacks.is_some())
        || (failure.is(ExceptionClass::ContentPolicyViolation)
            && context.content_policy_fallbacks.is_some())
        || failure
            .status_code
            .is_some_and(|status| !status_is_retryable(status) && status != 401 && status != 403)
        || failure.is(ExceptionClass::NotFound)
        || (failure.is(ExceptionClass::RateLimit)
            && context.healthy == 0
            && context
                .fallbacks
                .is_some_and(|fallbacks| !fallbacks.is_empty()))
        || ((failure.is(ExceptionClass::Authentication)
            || failure.is(ExceptionClass::PermissionDenied))
            && context.all <= 1)
        || context.healthy == 0;
    !no_retry
}

#[derive(Clone, Copy, Debug)]
pub struct Backoff {
    pub remaining: u32,
    pub num_retries: u32,
    pub healthy: usize,
    pub all: usize,
}

/// `_time_to_sleep_before_retry`: no wait while other deployments in a multi-deployment group
/// are healthy, otherwise `_calculate_retry_after`, which draws its jitter before deciding
/// between Retry-After and exponential backoff.
pub fn sleep_before_retry(
    failure: &Classified,
    backoff: Backoff,
    min_timeout: f64,
    tunables: &Tunables,
    random: &mut PythonRandom,
) -> f64 {
    let Backoff {
        remaining,
        num_retries,
        healthy,
        all,
    } = backoff;
    if all != 1 && healthy > 0 {
        return 0.0;
    }
    let jitter = tunables.jitter * random.random();
    let retry_after = failure.sleep_retry_after;
    if retry_after > 0 && retry_after <= 60 {
        return retry_after as f64 + jitter;
    }
    let exponent = i32::try_from(num_retries.saturating_sub(remaining)).unwrap_or(i32::MAX);
    let backoff = tunables.initial_retry_delay * 2f64.powi(exponent);
    backoff.max(min_timeout).min(tunables.max_retry_delay) + jitter
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Backoff, RetryContext, should_retry, sleep_before_retry};
    use crate::{
        failure::{Classified, ExceptionClass},
        random::PythonRandom,
        settings::{FallbackEntry, Fallbacks, Tunables},
    };

    fn failure(classes: &[ExceptionClass], status: Option<i64>) -> Classified {
        Classified {
            classes: classes.to_vec(),
            status_code: status,
            sleep_retry_after: -1,
            ..Classified::default()
        }
    }

    fn context<'a>(
        healthy: usize,
        all: usize,
        fallbacks: Option<&'a Fallbacks>,
    ) -> RetryContext<'a> {
        RetryContext {
            healthy,
            all,
            context_window_fallbacks: None,
            content_policy_fallbacks: None,
            fallbacks,
        }
    }

    #[rstest]
    #[case::server_error(&[ExceptionClass::InternalServer], Some(500), 1, 2, true)]
    #[case::bad_request(&[ExceptionClass::BadRequest], Some(400), 1, 2, false)]
    #[case::not_found_class(&[ExceptionClass::NotFound], None, 1, 2, false)]
    #[case::auth_with_alternatives(&[ExceptionClass::Authentication], Some(401), 1, 2, true)]
    #[case::auth_single_deployment(&[ExceptionClass::Authentication], Some(401), 1, 1, false)]
    #[case::nothing_healthy(&[ExceptionClass::InternalServer], Some(500), 0, 2, false)]
    #[case::status_less_router_error(&[], None, 1, 2, true)]
    fn should_retry_follows_python_rules(
        #[case] classes: &[ExceptionClass],
        #[case] status: Option<i64>,
        #[case] healthy: usize,
        #[case] all: usize,
        #[case] expected: bool,
    ) {
        assert_eq!(
            should_retry(&failure(classes, status), &context(healthy, all, None)),
            expected
        );
    }

    #[rstest]
    fn rate_limit_with_nothing_healthy_defers_to_fallbacks_only_when_they_exist() {
        let fallbacks = vec![FallbackEntry::Bare("other".into())];
        let rate_limit = failure(&[ExceptionClass::RateLimit], Some(429));
        assert!(!should_retry(&rate_limit, &context(0, 2, Some(&fallbacks))));
        assert!(!should_retry(&rate_limit, &context(0, 2, None)));
    }

    #[rstest]
    fn context_window_errors_wait_for_their_fallbacks() {
        let empty: Fallbacks = vec![];
        let error = failure(
            &[
                ExceptionClass::ContextWindowExceeded,
                ExceptionClass::BadRequest,
            ],
            None,
        );
        let with_list = RetryContext {
            context_window_fallbacks: Some(&empty),
            ..context(1, 2, None)
        };
        assert!(!should_retry(&error, &with_list));
        assert!(should_retry(&error, &context(1, 2, None)));
    }

    #[rstest]
    #[case::healthy_alternatives(1, 2, -1, 3, 3, 0.0)]
    #[case::retry_after_header(0, 2, 7, 3, 3, 7.0)]
    #[case::first_backoff(0, 2, -1, 3, 3, 0.5)]
    #[case::second_backoff(0, 2, -1, 2, 3, 1.0)]
    #[case::capped(0, 2, -1, 0, 9, 8.0)]
    #[case::single_deployment_ignores_health(1, 1, -1, 3, 3, 0.5)]
    fn sleep_matches_calculate_retry_after(
        #[case] healthy: usize,
        #[case] all: usize,
        #[case] retry_after: i64,
        #[case] remaining: u32,
        #[case] num_retries: u32,
        #[case] base: f64,
    ) {
        let tunables = Tunables {
            jitter: 0.0,
            ..Tunables::default()
        };
        let error = Classified {
            sleep_retry_after: retry_after,
            ..Classified::default()
        };
        let mut random = PythonRandom::seeded(1);
        assert_eq!(
            sleep_before_retry(
                &error,
                Backoff {
                    remaining,
                    num_retries,
                    healthy,
                    all
                },
                0.0,
                &tunables,
                &mut random
            ),
            base
        );
    }
}
