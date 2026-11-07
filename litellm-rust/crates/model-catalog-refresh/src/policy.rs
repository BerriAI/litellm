use std::time::Duration;

/// Main's defaults are 3 attempts, a 30 s max wait, a 5 s timeout, backoff base 2.0 and a jitter of 0.0 to 1.0 s
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub max_wait: Duration,
    pub timeout: Duration,
    pub backoff_base_seconds: f64,
    pub jitter_min_seconds: f64,
    pub jitter_max_seconds: f64,
}

/// Statuses main retries, every other error status ends the fetch
pub const RETRYABLE_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// Main's default `LITELLM_MODEL_COST_MAP_URL`
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

impl RetryPolicy {
    /// How long to wait after failed attempt number `attempt`: the server's `Retry-After` when it sent one, otherwise
    /// exponential backoff plus jitter, both capped at `max_wait`. `unit` is a uniform sample in `[0, 1)`
    pub fn wait(&self, attempt: u32, retry_after: Option<f64>, unit: f64) -> Duration {
        let seconds = retry_after.unwrap_or_else(|| {
            let jitter = self.jitter_min_seconds
                + unit * (self.jitter_max_seconds - self.jitter_min_seconds);
            self.backoff_base_seconds.powf(f64::from(attempt)) + jitter
        });
        Duration::from_secs_f64(seconds.min(self.max_wait.as_secs_f64()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const MAIN: RetryPolicy = RetryPolicy {
        max_attempts: 3,
        max_wait: Duration::from_secs(30),
        timeout: Duration::from_secs(5),
        backoff_base_seconds: 2.0,
        jitter_min_seconds: 0.0,
        jitter_max_seconds: 1.0,
    };

    #[rstest]
    #[case::retry_after_wins(1, Some(7.0), 0.9, 7.0)]
    #[case::retry_after_capped(1, Some(120.0), 0.0, 30.0)]
    #[case::infinite_retry_after_capped(1, Some(f64::INFINITY), 0.0, 30.0)]
    #[case::zero_retry_after(2, Some(0.0), 0.5, 0.0)]
    #[case::first_backoff(1, None, 0.25, 2.25)]
    #[case::second_backoff(2, None, 0.5, 4.5)]
    #[case::backoff_capped(5, None, 0.5, 30.0)]
    fn wait_follows_mains_retry_wait_seconds(
        #[case] attempt: u32,
        #[case] retry_after: Option<f64>,
        #[case] unit: f64,
        #[case] seconds: f64,
    ) {
        assert_eq!(
            MAIN.wait(attempt, retry_after, unit),
            Duration::from_secs_f64(seconds)
        );
    }
}
