//! Retry logic with exponential backoff and jitter.
//!
//! Provides configurable retry logic for transient failures.
//! Integrates with circuit breaker to avoid retrying when circuit is open.

use std::time::Duration;
use rand::Rng;

/// Retry configuration.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Base delay for exponential backoff.
    pub base_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
    /// Whether to add jitter to delays.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: true,
        }
    }
}

/// Calculate delay for a given retry attempt with exponential backoff and optional jitter.
pub fn calculate_delay(attempt: u32, config: &RetryConfig) -> Duration {
    // Exponential backoff: base_delay * 2^attempt
    let exponential = config.base_delay.as_millis() as u64 * 2u64.pow(attempt);
    let delay_ms = exponential.min(config.max_delay.as_millis() as u64);
    
    if config.jitter {
        // Add jitter: random value between 0 and delay_ms
        let jitter = rand::thread_rng().gen_range(0..=delay_ms);
        Duration::from_millis(jitter)
    } else {
        Duration::from_millis(delay_ms)
    }
}

/// Retry a function with exponential backoff.
///
/// Returns the result of the first successful attempt, or the last error if all attempts fail.
pub async fn retry_with_backoff<F, Fut, T, E>(
    config: &RetryConfig,
    mut f: F,
    is_retryable: fn(&E) -> bool,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut attempt = 0;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if !is_retryable(&err) || attempt >= config.max_retries {
                    return Err(err);
                }

                attempt += 1;
                let delay = calculate_delay(attempt, config);
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Check if an error is retryable.
///
/// Currently considers network errors and timeouts as retryable.
pub fn is_retryable_error(err: &litellm_core::CoreError) -> bool {
    matches!(
        err,
        litellm_core::CoreError::Network(_) | litellm_core::CoreError::Connect(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_delay_exponential() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: false,
        };

        assert_eq!(calculate_delay(0, &config), Duration::from_millis(100));
        assert_eq!(calculate_delay(1, &config), Duration::from_millis(200));
        assert_eq!(calculate_delay(2, &config), Duration::from_millis(400));
        assert_eq!(calculate_delay(3, &config), Duration::from_millis(800));
    }

    #[test]
    fn test_calculate_delay_with_max_cap() {
        let config = RetryConfig {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            jitter: false,
        };

        assert_eq!(calculate_delay(0, &config), Duration::from_millis(100));
        assert_eq!(calculate_delay(1, &config), Duration::from_millis(200));
        assert_eq!(calculate_delay(2, &config), Duration::from_millis(400));
        assert_eq!(calculate_delay(3, &config), Duration::from_millis(500)); // Capped
        assert_eq!(calculate_delay(4, &config), Duration::from_millis(500)); // Capped
    }

    #[test]
    fn test_calculate_delay_with_jitter() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: true,
        };

        let delay = calculate_delay(0, &config);
        assert!(delay <= Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let config = RetryConfig::default();
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async { Ok::<_, String>("success") }
        }, |_| true).await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts, 1);
    }

    #[tokio::test]
    async fn test_retry_success_after_retries() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            jitter: false,
        };
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async move {
                if attempts < 3 {
                    Err("temporary error")
                } else {
                    Ok("success")
                }
            }
        }, |_| true).await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts, 3);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let config = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            jitter: false,
        };
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async { Err::<(), _>("permanent error") }
        }, |_| true).await;

        assert!(result.is_err());
        assert_eq!(attempts, 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_non_retryable_returns_immediately() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            jitter: false,
        };
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async { Err::<(), _>("auth error") }
        }, |_| false).await;

        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_is_retryable_error() {
        assert!(is_retryable_error(&litellm_core::CoreError::Network("timeout".to_string())));
        assert!(is_retryable_error(&litellm_core::CoreError::Connect("connection refused".to_string())));
        assert!(!is_retryable_error(&litellm_core::CoreError::InvalidRequest("bad request".to_string())));
        assert!(!is_retryable_error(&litellm_core::CoreError::Auth("unauthorized".to_string())));
    }
}
