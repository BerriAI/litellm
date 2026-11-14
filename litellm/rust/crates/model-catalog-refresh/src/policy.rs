use std::time::Duration;

use backon::{BackoffBuilder, ExponentialBackoff, ExponentialBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub timeout: Duration,
    pub min_delay: Duration,
    pub max_delay: Duration,
}

pub const RETRYABLE_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// Main's default `LITELLM_MODEL_COST_MAP_URL`
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

impl RetryPolicy {
    pub(crate) fn backoff(&self) -> ExponentialBackoff {
        ExponentialBuilder::new()
            .with_min_delay(self.min_delay)
            .with_max_delay(self.max_delay)
            .with_max_times(self.max_attempts.saturating_sub(1) as usize)
            .with_jitter()
            .build()
    }

    pub(crate) fn delay(&self, backoff: Duration, retry_after: Option<Duration>) -> Duration {
        retry_after.map_or(backoff, |server| server.min(self.max_delay))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const POLICY: RetryPolicy = RetryPolicy {
        max_attempts: 3,
        timeout: Duration::from_secs(5),
        min_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(30),
    };

    #[rstest]
    #[case::backoff_without_retry_after(None, 4)]
    #[case::retry_after_wins(Some(7), 7)]
    #[case::retry_after_zero(Some(0), 0)]
    #[case::retry_after_capped(Some(120), 30)]
    fn the_server_retry_after_overrides_the_backoff_up_to_max_delay(
        #[case] retry_after: Option<u64>,
        #[case] seconds: u64,
    ) {
        assert_eq!(
            POLICY.delay(Duration::from_secs(4), retry_after.map(Duration::from_secs)),
            Duration::from_secs(seconds)
        );
    }

    #[rstest]
    #[case::one_attempt(1, 0)]
    #[case::three_attempts(3, 2)]
    #[case::zero_attempts(0, 0)]
    fn the_backoff_allows_one_retry_fewer_than_the_attempts(
        #[case] max_attempts: u32,
        #[case] retries: usize,
    ) {
        let policy = RetryPolicy {
            max_attempts,
            ..POLICY
        };
        assert_eq!(policy.backoff().count(), retries);
    }
}
