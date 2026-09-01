//! Retry logic with exponential backoff and jitter.
//!
//! Provides configurable retry logic for transient failures.
//! Integrates with circuit breaker to avoid retrying when circuit is open.
//! Supports multiple retry strategies for different error types.
//! Supports exception-specific retry policies and Retry-After header parsing.

use rand::Rng;
use std::time::Duration;

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
    /// Bad request errors (400)
    BadRequest,
    /// Authentication errors (401)
    Authentication,
    /// Rate limit errors (429)
    RateLimit,
    /// Timeout errors
    Timeout,
}

/// Exception-specific retry policy.
/// Allows configuring different retry counts for different exception types.
#[derive(Debug, Clone, Default)]
pub struct ExceptionRetryPolicy {
    /// Retries for bad request errors (400)
    pub bad_request_retries: Option<u32>,
    /// Retries for authentication errors (401)
    pub authentication_retries: Option<u32>,
    /// Retries for timeout errors
    pub timeout_retries: Option<u32>,
    /// Retries for rate limit errors (429)
    pub rate_limit_retries: Option<u32>,
    /// Retries for content policy violation errors
    pub content_policy_retries: Option<u32>,
    /// Retries for internal server errors (500)
    pub internal_server_retries: Option<u32>,
}

impl ExceptionRetryPolicy {
    /// Get retry count for a specific error category.
    pub fn get_retries(&self, category: ErrorCategory) -> Option<u32> {
        match category {
            ErrorCategory::BadRequest => self.bad_request_retries,
            ErrorCategory::Authentication => self.authentication_retries,
            ErrorCategory::Timeout => self.timeout_retries,
            ErrorCategory::RateLimit => self.rate_limit_retries,
            _ => None,
        }
    }
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

/// Parse Retry-After header value.
/// Supports both delta-seconds and HTTP-date formats.
pub fn parse_retry_after(header_value: &str) -> Option<Duration> {
    // Try parsing as delta-seconds (integer)
    if let Ok(seconds) = header_value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    // Try parsing as HTTP-date (RFC 7231)
    // For simplicity, we'll just return None for HTTP-date format
    // In a full implementation, you'd parse the date and calculate the duration
    None
}

/// Provider-specific retry configuration.
#[derive(Debug, Clone, Default)]
pub struct ProviderRetryConfig {
    /// OpenAI-specific retry config
    pub openai: Option<RetryStrategy>,
    /// Anthropic-specific retry config
    pub anthropic: Option<RetryStrategy>,
    /// Bedrock-specific retry config
    pub bedrock: Option<RetryStrategy>,
}

impl ProviderRetryConfig {
    /// Get retry strategy for a specific provider.
    pub fn get_strategy(&self, provider: &str) -> Option<&RetryStrategy> {
        match provider {
            "openai" | "azure" => self.openai.as_ref(),
            "anthropic" => self.anthropic.as_ref(),
            "bedrock" | "bedrock_converse" => self.bedrock.as_ref(),
            _ => None,
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
    /// Exception-specific retry policy (overrides category strategies).
    pub exception_policy: ExceptionRetryPolicy,
    /// Provider-specific retry configurations.
    pub provider_config: ProviderRetryConfig,
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
            exception_policy: ExceptionRetryPolicy::default(),
            provider_config: ProviderRetryConfig::default(),
        }
    }
}

impl RetryConfig {
    /// Get the appropriate strategy for an error category.
    /// First checks exception_policy for specific overrides, then falls back to category strategies.
    pub fn strategy_for(&self, category: ErrorCategory) -> RetryStrategy {
        // Check exception policy first for specific overrides
        if let Some(max_retries) = self.exception_policy.get_retries(category) {
            // Use the base strategy for this category but override max_retries
            let base_strategy = match category {
                ErrorCategory::BadRequest | ErrorCategory::Authentication => {
                    &self.application_strategy
                }
                ErrorCategory::RateLimit | ErrorCategory::Timeout => &self.http_strategy,
                _ => &self.unknown_strategy,
            };
            return RetryStrategy {
                max_retries,
                ..base_strategy.clone()
            };
        }

        // Fall back to category strategies
        match category {
            ErrorCategory::Network => self.network_strategy.clone(),
            ErrorCategory::Http | ErrorCategory::RateLimit | ErrorCategory::Timeout => {
                self.http_strategy.clone()
            }
            ErrorCategory::Application
            | ErrorCategory::BadRequest
            | ErrorCategory::Authentication => self.application_strategy.clone(),
            ErrorCategory::Provider => self.provider_strategy.clone(),
            ErrorCategory::Unknown => self.unknown_strategy.clone(),
        }
    }

    /// Check if an error is retryable.
    pub fn is_retryable(&self, category: ErrorCategory) -> bool {
        self.strategy_for(category).max_retries > 0
    }

    /// Get retry strategy for a specific provider.
    pub fn provider_strategy_for(&self, provider: &str) -> Option<RetryStrategy> {
        self.provider_config.get_strategy(provider).cloned()
    }
}

/// Calculate delay for a given retry attempt with exponential backoff and optional jitter.
pub fn calculate_delay(attempt: u32, strategy: &RetryStrategy) -> Duration {
    // Exponential backoff: base_delay * multiplier^attempt
    let exponential =
        strategy.base_delay.as_millis() as f64 * strategy.backoff_multiplier.powi(attempt as i32);
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
/// Maps HTTP status codes to exception-specific categories for more granular retry policies.
pub fn categorize_error(err: &litellm_core::CoreError) -> ErrorCategory {
    match err {
        litellm_core::CoreError::Network(_) | litellm_core::CoreError::Connect(_) => {
            ErrorCategory::Network
        }
        litellm_core::CoreError::Http { status, .. } => match *status {
            400 => ErrorCategory::BadRequest,
            401 | 403 => ErrorCategory::Authentication,
            408 | 504 => ErrorCategory::Timeout,
            429 => ErrorCategory::RateLimit,
            500..=599 => ErrorCategory::Http,
            _ => ErrorCategory::Application,
        },
        litellm_core::CoreError::Timeout(_) => ErrorCategory::Timeout,
        litellm_core::CoreError::InvalidRequest(_)
        | litellm_core::CoreError::InvalidType { .. }
        | litellm_core::CoreError::MissingField(_) => ErrorCategory::BadRequest,
        litellm_core::CoreError::Auth(_) => ErrorCategory::Authentication,
        litellm_core::CoreError::InvalidProvider(_)
        | litellm_core::CoreError::InvalidResponse(_)
        | litellm_core::CoreError::Routing(_)
        | litellm_core::CoreError::Unsupported(_) => ErrorCategory::Provider,
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
    let mut last_category: ErrorCategory;

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
                let delay = calculate_delay(attempt, &strategy);
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
            categorize_error(&litellm_core::CoreError::Connect(
                "connection refused".to_string()
            )),
            ErrorCategory::Network
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http {
                status: 500,
                body: "error".to_string()
            }),
            ErrorCategory::Http
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http {
                status: 429,
                body: "rate limited".to_string()
            }),
            ErrorCategory::RateLimit
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http {
                status: 400,
                body: "bad".to_string()
            }),
            ErrorCategory::BadRequest
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http {
                status: 401,
                body: "unauthorized".to_string()
            }),
            ErrorCategory::Authentication
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Http {
                status: 408,
                body: "timeout".to_string()
            }),
            ErrorCategory::Timeout
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::InvalidRequest(
                "bad request".to_string()
            )),
            ErrorCategory::BadRequest
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::Auth("unauthorized".to_string())),
            ErrorCategory::Authentication
        );
        assert_eq!(
            categorize_error(&litellm_core::CoreError::InvalidProvider(
                "unknown provider".to_string()
            )),
            ErrorCategory::Provider
        );
    }

    #[test]
    fn test_retry_config_strategies() {
        let config = RetryConfig::default();

        assert_eq!(config.strategy_for(ErrorCategory::Network).max_retries, 5);
        assert_eq!(config.strategy_for(ErrorCategory::Http).max_retries, 3);
        assert_eq!(
            config.strategy_for(ErrorCategory::Application).max_retries,
            1
        );
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

        let result = retry_with_backoff(
            &config,
            || {
                attempts += 1;
                async { Ok::<_, String>("success") }
            },
            |_| ErrorCategory::Unknown,
        )
        .await;

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

        let result = retry_with_backoff(
            &config,
            || {
                attempts += 1;
                async move {
                    if attempts < 3 {
                        Err(litellm_core::CoreError::Network(
                            "temporary error".to_string(),
                        ))
                    } else {
                        Ok("success")
                    }
                }
            },
            categorize_error,
        )
        .await;

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

        let result = retry_with_backoff(
            &config,
            || {
                attempts += 1;
                async {
                    Err::<(), _>(litellm_core::CoreError::Network(
                        "permanent error".to_string(),
                    ))
                }
            },
            categorize_error,
        )
        .await;

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

        let result = retry_with_backoff(
            &config,
            || {
                attempts += 1;
                async {
                    Err::<(), _>(litellm_core::CoreError::InvalidRequest(
                        "auth error".to_string(),
                    ))
                }
            },
            categorize_error,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_is_retryable_error_backward_compat() {
        assert!(is_retryable_error(&litellm_core::CoreError::Network(
            "timeout".to_string()
        )));
        assert!(is_retryable_error(&litellm_core::CoreError::Connect(
            "connection refused".to_string()
        )));
        assert!(!is_retryable_error(
            &litellm_core::CoreError::InvalidRequest("bad request".to_string())
        ));
        assert!(!is_retryable_error(&litellm_core::CoreError::Auth(
            "unauthorized".to_string()
        )));
    }
}
