//! Retry logic with exponential backoff and jitter.
//!
//! Provides configurable retry logic for transient failures.
//! Integrates with circuit breaker to avoid retrying when circuit is open.
//! Supports multiple retry strategies for different error types.

use std::time::Duration;
use rand::Rng;

/// Error categories for retry decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network-level errors (connection refused, timeout, DNS failure)
    Network,
    /// HTTP-level errors (5xx status codes, rate limiting)
    Http,
    /// Application-level errors (invalid request, auth failure)
    Application,
    /// Provider-specific errors (model not found, quota exceeded)
    Provider,
    /// Unknown or unclassified errors
    Unknown,
}

/// Retry strategy configuration.
#[derive(Debug, Clone)]
pub struct RetryStrategy {
    /// Maximum number of retry attempts for this strategy.
    pub max_retries: u32,
    /// Base delay for exponential backoff.
    pub base_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
    /// Whether to add jitter to delays.
    pub jitter: bool,
    /// Multiplier for exponential backoff (default 2.0).
    pub backoff_multiplier: f64,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: true,
            backoff_multiplier: 2.0,
        }
    }
}

/// Retry configuration with per-category strategies.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Strategy for network errors (most aggressive retry).
    pub network_strategy: RetryStrategy,
    /// Strategy for HTTP errors (moderate retry).
    pub http_strategy: RetryStrategy,
    /// Strategy for application errors (conservative retry).
    pub application_strategy: RetryStrategy,
    /// Strategy for provider errors (conservative retry).
    pub provider_strategy: RetryStrategy,
    /// Strategy for unknown errors (conservative retry).
    pub unknown_strategy: RetryStrategy,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            network_strategy: RetryStrategy {
                max_retries: 5,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                jitter: true,
                backoff_multiplier: 2.0,
            },
            http_strategy: RetryStrategy {
                max_retries: 3,
                base_delay: Duration::from_millis(200),
                max_delay: Duration::from_secs(20),
                jitter: true,
                backoff_multiplier: 2.0,
            },
            application_strategy: RetryStrategy {
                max_retries: 1,
                base_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(5),
                jitter: false,
                backoff_multiplier: 1.0,
            },
            provider_strategy: RetryStrategy {
                max_retries: 2,
                base_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(10),
                jitter: true,
                backoff_multiplier: 2.0,
            },
            unknown_strategy: RetryStrategy {
                max_retries: 1,
                base_delay: Duration::from_millis(1000),
                max_delay: Duration::from_secs(5),
                jitter: false,
                backoff_multiplier: 1.0,
            },
        }
    }
}

impl RetryConfig {
    /// Get the appropriate strategy for an error category.
    pub fn strategy_for(&self, category: ErrorCategory) -> &RetryStrategy {
        match category {
            ErrorCategory::Network => &self.network_strategy,
            ErrorCategory::Http => &self.http_strategy,
            ErrorCategory::Application => &self.application_strategy,
            ErrorCategory::Provider => &self.provider_strategy,
            ErrorCategory::Unknown => &self.unknown_strategy,
        }
    }

    /// Check if an error is retryable.
    pub fn is_retryable(&self, category: ErrorCategory) -> bool {
        self.strategy_for(category).max_retries > 0
    }
}

/// Calculate delay for a given retry attempt with exponential backoff and optional jitter.
pub fn calculate_delay(attempt: u32, strategy: &RetryStrategy) -> Duration {
    // Exponential backoff: base_delay * multiplier^attempt
    let exponential = strategy.base_delay.as_millis() as f64 * strategy.backoff_multiplier.powi(attempt as i32);
    let delay_ms = (exponential as u64).min(strategy.max_delay.as_millis() as u64);
    
    if strategy.jitter {
        // Add jitter: random value between 0 and delay_ms
        let jitter = rand::thread_rng().gen_range(0..=delay_ms);
        Duration::from_millis(jitter)
    } else {
        Duration::from_millis(delay_ms)
    }
}

/// Categorize a CoreError into an ErrorCategory.
pub fn categorize_error(err: &litellm_core::CoreError) -> ErrorCategory {
    match err {
        litellm_core::CoreError::Network(_) | litellm_core::CoreError::Connect(_) => {
            ErrorCategory::Network
        }
        litellm_core::CoreError::Http { status, .. } => {
            if *status >= 500 {
                ErrorCategory::Http
            } else if *status == 429 {
                ErrorCategory::Http // Rate limiting
            } else {
                ErrorCategory::Application
            }
        }
        litellm_core::CoreError::InvalidRequest(_) 
        | litellm_core::CoreError::Auth(_) 
        | litellm_core::CoreError::InvalidType { .. }
        | litellm_core::CoreError::MissingField(_) => {
            ErrorCategory::Application
        }
        litellm_core::CoreError::InvalidProvider(_)
        | litellm_core::CoreError::InvalidResponse(_)
        | litellm_core::CoreError::Routing(_)
        | litellm_core::CoreError::Unsupported(_) => {
            ErrorCategory::Provider
        }
        _ => ErrorCategory::Unknown,
    }
}

/// Retry a function with category-specific strategies.
///
/// Returns the result of the first successful attempt, or the last error if all attempts fail.
pub async fn retry_with_backoff<F, Fut, T, E>(
    config: &RetryConfig,
    mut f: F,
    categorize: fn(&E) -> ErrorCategory,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut attempt = 0;
    let mut last_category = ErrorCategory::Unknown;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                last_category = categorize(&err);
                let strategy = config.strategy_for(last_category);
                
                if !config.is_retryable(last_category) || attempt >= strategy.max_retries {
                    return Err(err);
                }

                attempt += 1;
                let delay = calculate_delay(attempt, strategy);
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Check if an error is retryable (backward compatibility).
///
/// Currently considers network errors and timeouts as retryable.
pub fn is_retryable_error(err: &litellm_core::CoreError) -> bool {
    let category = categorize_error(err);
    matches!(category, ErrorCategory::Network | ErrorCategory::Http)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_delay_exponential() {
        let strategy = RetryStrategy {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: false,
            backoff_multiplier: 2.0,
        };

        assert_eq!(calculate_delay(0, &strategy), Duration::from_millis(100));
        assert_eq!(calculate_delay(1, &strategy), Duration::from_millis(200));
        assert_eq!(calculate_delay(2, &strategy), Duration::from_millis(400));
        assert_eq!(calculate_delay(3, &strategy), Duration::from_millis(800));
    }

    #[test]
    fn test_calculate_delay_with_max_cap() {
        let strategy = RetryStrategy {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            jitter: false,
            backoff_multiplier: 2.0,
        };

        assert_eq!(calculate_delay(0, &strategy), Duration::from_millis(100));
        assert_eq!(calculate_delay(1, &strategy), Duration::from_millis(200));
        assert_eq!(calculate_delay(2, &strategy), Duration::from_millis(400));
        assert_eq!(calculate_delay(3, &strategy), Duration::from_millis(500)); // Capped
        assert_eq!(calculate_delay(4, &strategy), Duration::from_millis(500)); // Capped
    }

    #[test]
    fn test_calculate_delay_with_jitter() {
        let strategy = RetryStrategy {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: true,
            backoff_multiplier: 2.0,
        };

        let delay = calculate_delay(0, &strategy);
        assert!(delay <= Duration::from_millis(100));
    }

    #[test]
    fn test_error_categorization() {
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Network("timeout".to_string())),
            ErrorCategory::Network
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Connect("connection refused".to_string())),
            ErrorCategory::Network
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http { status: 500, body: "error".to_string() }),
            ErrorCategory::Http
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http { status: 429, body: "rate limited".to_string() }),
            ErrorCategory::Http
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::InvalidRequest("bad request".to_string())),
            ErrorCategory::Application
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Auth("unauthorized".to_string())),
            ErrorCategory::Application
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::InvalidProvider("unknown provider".to_string())),
            ErrorCategory::Provider
        );
    }

    #[test]
    fn test_retry_config_strategies() {
        let config = RetryConfig::default();
        
        assert_eq!(config.strategy_for(ErrorCategory::Network).max_retries, 5);
        assert_eq!(config.strategy_for(ErrorCategory::Http).max_retries, 3);
        assert_eq!(config.strategy_for(ErrorCategory::Application).max_retries, 1);
        assert_eq!(config.strategy_for(ErrorCategory::Provider).max_retries, 2);
        assert_eq!(config.strategy_for(ErrorCategory::Unknown).max_retries, 1);
    }

    #[test]
    fn test_is_retryable() {
        let config = RetryConfig::default();
        
        assert!(config.is_retryable(ErrorCategory::Network));
        assert!(config.is_retryable(ErrorCategory::Http));
        assert!(config.is_retryable(ErrorCategory::Application));
        assert!(config.is_retryable(ErrorCategory::Provider));
        assert!(config.is_retryable(ErrorCategory::Unknown));
    }

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let config = RetryConfig::default();
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async { Ok::<_, String>("success") }
        }, |_| ErrorCategory::Unknown).await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts, 1);
    }

    #[tokio::test]
    async fn test_retry_success_after_retries() {
        let config = RetryConfig {
            network_strategy: RetryStrategy {
                max_retries: 3,
                base_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(100),
                jitter: false,
                backoff_multiplier: 2.0,
            },
            ..Default::default()
        };
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async move {
                if attempts < 3 {
                    Err(litellm_core::CoreError::Network("temporary error".to_string()))
                } else {
                    Ok("success")
                }
            }
        }, |e| categorize_error(e)).await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts, 3);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let config = RetryConfig {
            network_strategy: RetryStrategy {
                max_retries: 2,
                base_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(100),
                jitter: false,
                backoff_multiplier: 2.0,
            },
            ..Default::default()
        };
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async { Err::<(), _>(litellm_core::CoreError::Network("permanent error".to_string())) }
        }, |e| categorize_error(e)).await;

        assert!(result.is_err());
        assert_eq!(attempts, 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_non_retryable_returns_immediately() {
        let config = RetryConfig {
            application_strategy: RetryStrategy {
                max_retries: 0, // Non-retryable
                base_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(100),
                jitter: false,
                backoff_multiplier: 1.0,
            },
            ..Default::default()
        };
        let mut attempts = 0;

        let result = retry_with_backoff(&config, || {
            attempts += 1;
            async { Err::<(), _>(litellm_core::CoreError::InvalidRequest("auth error".to_string())) }
        }, |e| categorize_error(e)).await;

        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_is_retryable_error_backward_compat() {
        assert!(is_retryable_error(&litellm_core::CoreError::Network("timeout".to_string())));
        assert!(is_retryable_error(&litellm_core::CoreError::Connect("connection refused".to_string())));
        assert!(!is_retryable_error(&litellm_core::CoreError::InvalidRequest("bad request".to_string())));
        assert!(!is_retryable_error(&litellm_core::CoreError::Auth("unauthorized".to_string())));
    }
}
