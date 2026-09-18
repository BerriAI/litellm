//! What the Python router does after a failed attempt, as one pure table. `decide` reads
//! the failure and a [`Situation`] the loop assembled and returns the [`Decision`] the loop
//! carries out; nothing here touches a machine, a clock or a host.

use std::time::Duration;

use litellm_callbacks::failure::FailureClass;

use crate::plan::{Deployment, Retries, RetryPolicy};

/// What the loop does after one attempt fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Another attempt in the same group after `backoff`. `skip_failed` keeps the
    /// deployment that just failed out of the next pick; a lone deployment is never
    /// skipped, since the retry would have nothing left to pick.
    Retry {
        skip_failed: bool,
        backoff: Duration,
    },
    /// The group is done with; the next group for this failure, if any, is tried.
    Fallback,
    /// No further attempt anywhere; the failure is the call's.
    Stop,
}

/// Whether a group could follow the current one for each kind of failure. All true when
/// the host resolves groups, since the router cannot know what it will answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainsConfigured {
    pub generic: bool,
    pub context_window: bool,
    pub content_policy: bool,
}

impl ChainsConfigured {
    pub const ALL: Self = Self {
        generic: true,
        context_window: true,
        content_policy: true,
    };
    pub const NONE: Self = Self {
        generic: false,
        context_window: false,
        content_policy: false,
    };

    /// Whether any group follows for this failure: Python tries the class's own chain
    /// when it is configured and the generic chain otherwise.
    pub const fn reaches(self, class: FailureClass) -> bool {
        match class {
            FailureClass::ContextWindow => self.context_window || self.generic,
            FailureClass::ContentPolicy => self.content_policy || self.generic,
            _ => self.generic,
        }
    }
}

/// Everything Python's retry and fallback code reads besides the exception itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Situation {
    pub class: FailureClass,
    pub retry_after: Option<Duration>,
    /// Retries already spent in this group.
    pub retries_used: u32,
    /// The deployment that failed, with its own retry count.
    pub failed: Deployment,
    /// Deployments in the group; Python's `all_deployments`.
    pub group_size: usize,
    /// Deployments the picker may still choose, not cooled down and not skipped, counting
    /// the failed one unless something already took it out; Python's `healthy_deployments`.
    pub available: usize,
    pub chains: ChainsConfigured,
}

/// How many retries the failed deployment gets, and whether a per-class policy chose the
/// number (Python's `_retry_policy_applies`, which then skips every other check).
pub fn retries_allowed(
    policy: &RetryPolicy,
    failed: &Deployment,
    class: FailureClass,
) -> (u32, bool) {
    let base = match policy.retries {
        Retries::Request(retries) => retries,
        Retries::Default(retries) => failed.retries.unwrap_or(retries),
    };
    let request_set_zero = policy.retries == Retries::Request(0);
    if request_set_zero || !policy.has_class_policy() {
        return (base, false);
    }
    match policy.class_retries(class) {
        Some(retries) => (retries, true),
        None => (base, false),
    }
}

/// Python's decision after a failed attempt: `should_retry_this_error`, then the retry
/// count, then `_time_to_sleep_before_retry`, then the fallback dispatch. `jitter` is the
/// jitter already sampled for this wait.
pub fn decide(situation: &Situation, policy: &RetryPolicy, jitter: Duration) -> Decision {
    let (allowed, policy_applies) = retries_allowed(policy, &situation.failed, situation.class);
    let retryable = policy_applies || should_retry(situation);
    if retryable && situation.retries_used < allowed {
        return Decision::Retry {
            skip_failed: !situation.class.retryable_status() && situation.group_size > 1,
            backoff: backoff(situation, policy, jitter),
        };
    }
    if situation.chains.reaches(situation.class) {
        Decision::Fallback
    } else {
        Decision::Stop
    }
}

/// `Router.should_retry_this_error`, as a predicate instead of a raise.
fn should_retry(situation: &Situation) -> bool {
    let class = situation.class;
    let chains = situation.chains;
    if class == FailureClass::ContextWindow && chains.context_window {
        return false;
    }
    if class == FailureClass::ContentPolicy && chains.content_policy {
        return false;
    }
    if !class.retryable_status() && class != FailureClass::Authentication {
        return false;
    }
    if class == FailureClass::RateLimited && situation.available == 0 && chains.generic {
        return false;
    }
    if class == FailureClass::Authentication && situation.group_size <= 1 {
        return false;
    }
    situation.available > 0
}

/// `Router._time_to_sleep_before_retry` over `litellm._calculate_retry_after`: nothing
/// when another deployment in a multi-deployment group is available, else the provider's
/// `Retry-After` when reasonable, else exponential backoff between the floor and the cap.
fn backoff(situation: &Situation, policy: &RetryPolicy, jitter: Duration) -> Duration {
    if situation.group_size > 1 && situation.available > 0 {
        return Duration::ZERO;
    }
    if let Some(retry_after) = situation.retry_after
        && retry_after > Duration::ZERO
        && retry_after <= policy.max_retry_after
    {
        return retry_after + jitter;
    }
    let exponent = situation.retries_used.min(16);
    let base = policy
        .initial_backoff
        .checked_mul(1 << exponent)
        .unwrap_or(policy.max_backoff);
    base.max(policy.min_backoff).min(policy.max_backoff) + jitter
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::plan::DeploymentId;

    const ALL: ChainsConfigured = ChainsConfigured::ALL;
    const NONE: ChainsConfigured = ChainsConfigured::NONE;
    const GENERIC: ChainsConfigured = ChainsConfigured {
        generic: true,
        context_window: false,
        content_policy: false,
    };

    fn deployment() -> Deployment {
        Deployment::new(DeploymentId(1))
    }

    fn situation(
        class: FailureClass,
        group_size: usize,
        available: usize,
        chains: ChainsConfigured,
    ) -> Situation {
        Situation {
            class,
            retry_after: None,
            retries_used: 0,
            failed: deployment(),
            group_size,
            available,
            chains,
        }
    }

    fn python(retries: u32) -> RetryPolicy {
        RetryPolicy::python(Retries::Default(retries))
    }

    fn ms(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    /// `Router.should_retry_this_error`, row by row.
    #[rstest]
    #[case::context_window_with_its_chain(FailureClass::ContextWindow, 2, 2, ALL, false)]
    #[case::context_window_is_a_400_without_it(FailureClass::ContextWindow, 2, 2, NONE, false)]
    #[case::content_policy_with_its_chain(FailureClass::ContentPolicy, 2, 2, ALL, false)]
    #[case::rate_limit_on_a_lone_deployment(FailureClass::RateLimited, 1, 1, NONE, true)]
    #[case::rate_limit_with_nothing_healthy_and_fallbacks(
        FailureClass::RateLimited,
        2,
        0,
        GENERIC,
        false
    )]
    #[case::rate_limit_with_nothing_healthy_and_no_fallbacks(
        FailureClass::RateLimited,
        2,
        0,
        NONE,
        false
    )]
    #[case::rate_limit_with_a_healthy_sibling(FailureClass::RateLimited, 2, 1, NONE, true)]
    #[case::auth_error_on_a_lone_deployment(FailureClass::Authentication, 1, 1, NONE, false)]
    #[case::auth_error_with_siblings(FailureClass::Authentication, 2, 2, NONE, true)]
    #[case::not_found_never(FailureClass::NotFound, 3, 3, ALL, false)]
    #[case::bad_request_never(FailureClass::BadRequest, 2, 2, ALL, false)]
    #[case::server_error_with_healthy_deployments(FailureClass::InternalServer, 2, 2, NONE, true)]
    #[case::server_error_with_nothing_healthy(FailureClass::InternalServer, 2, 0, NONE, false)]
    #[case::connection_error_has_no_status_to_reject(FailureClass::Connection, 1, 1, NONE, true)]
    #[case::timeout_is_retried(FailureClass::Timeout, 1, 1, NONE, true)]
    fn should_retry_this_error(
        #[case] class: FailureClass,
        #[case] group_size: usize,
        #[case] available: usize,
        #[case] chains: ChainsConfigured,
        #[case] expected: bool,
    ) {
        assert_eq!(
            should_retry(&situation(class, group_size, available, chains)),
            expected
        );
    }

    /// `async_function_with_retries`: request, deployment and router counts, then
    /// `get_num_retries_from_retry_policy` walking the exception's MRO.
    #[rstest]
    #[case::router_default(Retries::Default(2), None, &[], None, FailureClass::RateLimited, (2, false))]
    #[case::deployment_beats_router_default(Retries::Default(2), Some(5), &[], None, FailureClass::RateLimited, (5, false))]
    #[case::request_beats_deployment(Retries::Request(2), Some(5), &[], None, FailureClass::RateLimited, (2, false))]
    #[case::request_zero_disables_the_policy(Retries::Request(0), None, &[(FailureClass::RateLimited, 3)], None, FailureClass::RateLimited, (0, false))]
    #[case::class_count_applies(Retries::Default(2), None, &[(FailureClass::RateLimited, 3)], None, FailureClass::RateLimited, (3, true))]
    #[case::context_window_walks_to_bad_request(Retries::Default(2), None, &[(FailureClass::BadRequest, 1)], None, FailureClass::ContextWindow, (1, true))]
    #[case::unlisted_class_without_default_keeps_the_count(Retries::Default(2), None, &[(FailureClass::RateLimited, 3)], None, FailureClass::NotFound, (2, false))]
    #[case::unlisted_class_takes_the_policy_default(Retries::Default(2), None, &[], Some(4), FailureClass::NotFound, (4, true))]
    #[case::listed_class_beats_the_policy_default(Retries::Default(2), None, &[(FailureClass::RateLimited, 3)], Some(4), FailureClass::RateLimited, (3, true))]
    fn retries_come_from_the_request_the_deployment_the_router_or_the_policy(
        #[case] retries: Retries,
        #[case] deployment_retries: Option<u32>,
        #[case] class_retries: &[(FailureClass, u32)],
        #[case] class_default: Option<u32>,
        #[case] class: FailureClass,
        #[case] expected: (u32, bool),
    ) {
        let policy = RetryPolicy {
            retries,
            class_retries: class_retries.to_vec(),
            class_default,
            ..RetryPolicy::python(retries)
        };
        let failed = Deployment {
            retries: deployment_retries,
            ..deployment()
        };
        assert_eq!(retries_allowed(&policy, &failed, class), expected);
    }

    fn retry(skip_failed: bool, backoff: u64) -> Decision {
        Decision::Retry {
            skip_failed,
            backoff: ms(backoff),
        }
    }

    /// The whole decision, with Python's timing constants and 100ms of sampled jitter.
    #[rstest]
    #[case::lone_rate_limit_backs_off_from_half_a_second(
        FailureClass::RateLimited,
        0,
        None,
        1,
        1,
        NONE,
        retry(false, 600)
    )]
    #[case::second_retry_doubles(
        FailureClass::RateLimited,
        1,
        None,
        1,
        1,
        NONE,
        retry(false, 1100)
    )]
    #[case::retries_exhausted_fall_back(
        FailureClass::RateLimited,
        2,
        None,
        1,
        1,
        GENERIC,
        Decision::Fallback
    )]
    #[case::retries_exhausted_without_chains_stop(
        FailureClass::RateLimited,
        2,
        None,
        1,
        1,
        NONE,
        Decision::Stop
    )]
    #[case::a_healthy_sibling_means_no_wait(
        FailureClass::RateLimited,
        0,
        Some(3_000),
        2,
        2,
        NONE,
        retry(false, 0)
    )]
    #[case::retry_after_is_obeyed_alone(
        FailureClass::RateLimited,
        0,
        Some(3_000),
        1,
        1,
        NONE,
        retry(false, 3_100)
    )]
    #[case::retry_after_beyond_a_minute_is_ignored(
        FailureClass::RateLimited,
        0,
        Some(61_000),
        1,
        1,
        NONE,
        retry(false, 600)
    )]
    #[case::zero_retry_after_is_ignored(
        FailureClass::RateLimited,
        0,
        Some(0),
        1,
        1,
        NONE,
        retry(false, 600)
    )]
    #[case::auth_error_with_siblings_skips_the_failed_one(
        FailureClass::Authentication,
        0,
        None,
        2,
        2,
        NONE,
        retry(true, 0)
    )]
    #[case::auth_error_alone_falls_back(
        FailureClass::Authentication,
        0,
        None,
        1,
        1,
        ALL,
        Decision::Fallback
    )]
    #[case::auth_error_alone_without_chains_stops(
        FailureClass::Authentication,
        0,
        None,
        1,
        1,
        NONE,
        Decision::Stop
    )]
    #[case::context_window_takes_its_chain_at_once(
        FailureClass::ContextWindow,
        0,
        None,
        2,
        2,
        ALL,
        Decision::Fallback
    )]
    #[case::bad_request_with_only_a_context_window_chain_stops(FailureClass::BadRequest, 0, None, 2, 2, ChainsConfigured { generic: false, context_window: true, content_policy: false }, Decision::Stop)]
    #[case::nothing_healthy_in_a_pair_waits_like_a_lone_deployment(
        FailureClass::InternalServer,
        0,
        None,
        2,
        0,
        NONE,
        Decision::Stop
    )]
    fn decide_matches_python(
        #[case] class: FailureClass,
        #[case] retries_used: u32,
        #[case] retry_after: Option<u64>,
        #[case] group_size: usize,
        #[case] available: usize,
        #[case] chains: ChainsConfigured,
        #[case] expected: Decision,
    ) {
        let situation = Situation {
            retries_used,
            retry_after: retry_after.map(ms),
            ..situation(class, group_size, available, chains)
        };
        assert_eq!(decide(&situation, &python(2), ms(100)), expected);
    }

    #[test]
    fn a_class_policy_retries_whatever_the_status_and_waits_when_nothing_is_healthy() {
        let policy = RetryPolicy {
            class_retries: vec![
                (FailureClass::BadRequest, 1),
                (FailureClass::InternalServer, 2),
            ],
            ..python(0)
        };
        assert_eq!(
            decide(
                &situation(FailureClass::BadRequest, 2, 2, NONE),
                &policy,
                ms(100)
            ),
            retry(true, 0)
        );
        assert_eq!(
            decide(
                &situation(FailureClass::InternalServer, 2, 0, NONE),
                &policy,
                ms(100)
            ),
            retry(false, 600)
        );
        let exhausted = Situation {
            retries_used: 1,
            ..situation(FailureClass::BadRequest, 2, 2, NONE)
        };
        assert_eq!(decide(&exhausted, &policy, ms(100)), Decision::Stop);
    }

    #[test]
    fn backoff_is_floored_capped_and_never_overflows() {
        let policy = RetryPolicy {
            min_backoff: ms(2_000),
            ..python(10)
        };
        let waits: Vec<Duration> = (0..7)
            .map(|used| {
                let situation = Situation {
                    retries_used: used,
                    ..situation(FailureClass::RateLimited, 1, 1, NONE)
                };
                backoff(&situation, &policy, Duration::ZERO)
            })
            .collect();
        assert_eq!(
            waits,
            [2_000, 2_000, 2_000, 4_000, 8_000, 8_000, 8_000].map(ms)
        );
        let deep = Situation {
            retries_used: u32::MAX,
            ..situation(FailureClass::RateLimited, 1, 1, NONE)
        };
        assert_eq!(backoff(&deep, &policy, Duration::ZERO), ms(8_000));
    }
}
