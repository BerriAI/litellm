const MAX_WAIT_SECONDS: f64 = 30.0;
const BACKOFF_BASE_SECONDS: f64 = 2.0;

/// What follows a retryable failure of attempt `attempt` out of `max_attempts`
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AfterRetryable {
    GiveUp,
    Wait { seconds: f64 },
}

/// A usable `Retry-After` is waited, capped at 30 s. Otherwise `2^attempt` plus `jitter()`, capped the same, so
/// `jitter` runs only when there is no usable `Retry-After`
pub fn after_retryable(
    attempt: i64,
    max_attempts: i64,
    retry_after_seconds: Option<f64>,
    jitter: impl FnOnce() -> f64,
) -> AfterRetryable {
    if attempt == max_attempts {
        return AfterRetryable::GiveUp;
    }
    let seconds = match retry_after_seconds.filter(|seconds| *seconds >= 0.0) {
        Some(retry_after) => retry_after.min(MAX_WAIT_SECONDS),
        None => (backoff(attempt) + jitter()).min(MAX_WAIT_SECONDS),
    };
    AfterRetryable::Wait { seconds }
}

fn backoff(attempt: i64) -> f64 {
    i32::try_from(attempt).map_or(f64::INFINITY, |exponent| {
        BACKOFF_BASE_SECONDS.powi(exponent)
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::last_attempt(3, 3, Some(1.0), AfterRetryable::GiveUp)]
    #[case::retry_after(1, 3, Some(4.5), AfterRetryable::Wait { seconds: 4.5 })]
    #[case::retry_after_capped(1, 3, Some(120.0), AfterRetryable::Wait { seconds: 30.0 })]
    #[case::infinite_retry_after(1, 3, Some(f64::INFINITY), AfterRetryable::Wait { seconds: 30.0 })]
    #[case::zero_retry_after(1, 3, Some(0.0), AfterRetryable::Wait { seconds: 0.0 })]
    #[case::negative_retry_after_backs_off(1, 3, Some(-1.0), AfterRetryable::Wait { seconds: 2.25 })]
    #[case::nan_retry_after_backs_off(1, 3, Some(f64::NAN), AfterRetryable::Wait { seconds: 2.25 })]
    #[case::first_backoff(1, 3, None, AfterRetryable::Wait { seconds: 2.25 })]
    #[case::second_backoff(2, 3, None, AfterRetryable::Wait { seconds: 4.25 })]
    #[case::backoff_capped(5, 9, None, AfterRetryable::Wait { seconds: 30.0 })]
    #[case::past_the_last_attempt_still_waits(4, 3, None, AfterRetryable::Wait { seconds: 16.25 })]
    fn waits_match_main(
        #[case] attempt: i64,
        #[case] max_attempts: i64,
        #[case] retry_after: Option<f64>,
        #[case] expected: AfterRetryable,
    ) {
        assert_eq!(
            after_retryable(attempt, max_attempts, retry_after, || 0.25),
            expected
        );
    }

    #[rstest]
    #[case::retry_after(Some(1.0), false)]
    #[case::no_retry_after(None, true)]
    #[case::unusable_retry_after(Some(-1.0), true)]
    #[case::give_up(None, false)]
    fn jitter_is_drawn_only_for_a_backoff(#[case] retry_after: Option<f64>, #[case] drawn: bool) {
        let mut draws = 0;
        let max_attempts = if drawn || retry_after.is_some() { 3 } else { 1 };
        let _ = after_retryable(1, max_attempts, retry_after, || {
            draws += 1;
            0.0
        });
        assert_eq!(draws, usize::from(drawn));
    }
}
