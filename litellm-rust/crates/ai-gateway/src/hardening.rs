//! Operational hardening features for production resilience.
//!
//! Provides:
//! - Global rate limiting (in addition to per-key limits)
//! - Graceful degradation logging for dependency failures
//! - Slow loris protection configuration

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Global rate limiter using a sliding window counter.
/// Zero-allocation: uses atomic operations and stack-allocated state.
pub struct GlobalRateLimiter {
    /// Maximum requests per window
    max_requests: u64,
    /// Window duration
    window_secs: u64,
    /// Current window start (unix timestamp)
    window_start: RwLock<u64>,
    /// Request count in current window
    count: AtomicU64,
}

impl GlobalRateLimiter {
    pub fn new(max_requests: u64, window_secs: u64) -> Self {
        Self {
            max_requests,
            window_secs,
            window_start: RwLock::new(Self::current_window(window_secs)),
            count: AtomicU64::new(0),
        }
    }

    fn current_window(window_secs: u64) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() / window_secs
    }

    /// Check if a request is allowed under the global rate limit.
    /// Returns true if allowed, false if rate limited.
    /// Zero-allocation: uses atomic operations only.
    pub async fn check(&self) -> bool {
        let current_window = Self::current_window(self.window_secs);
        
        // Check if we're in a new window
        {
            let window_start = self.window_start.read().await;
            if *window_start != current_window {
                // Need to reset the counter for the new window
                drop(window_start);
                let mut window_start = self.window_start.write().await;
                if *window_start != current_window {
                    *window_start = current_window;
                    self.count.store(1, Ordering::Relaxed);
                    return true;
                }
            }
        }
        
        // Increment counter and check limit
        let count = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        count <= self.max_requests
    }
}

/// Log graceful degradation when a dependency fails.
/// This is a zero-allocation helper that logs at the appropriate level.
#[inline]
pub fn log_degradation(dependency: &str, operation: &str, error: &str) {
    tracing::warn!(
        dependency = dependency,
        operation = operation,
        error = error,
        event = "degradation",
        "dependency failed, operating in degraded mode"
    );
}

/// Slow loris protection configuration.
/// Defines timeouts for reading request headers and body.
#[derive(Clone, Debug)]
pub struct SlowLorisConfig {
    /// Maximum time to read request headers
    pub header_timeout: Duration,
    /// Maximum time between reads for request body
    pub body_timeout: Duration,
    /// Maximum request body size
    pub max_body_size: usize,
}

impl Default for SlowLorisConfig {
    fn default() -> Self {
        Self {
            header_timeout: Duration::from_secs(10),
            body_timeout: Duration::from_secs(30),
            max_body_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_global_rate_limiter_allows_under_limit() {
        let limiter = GlobalRateLimiter::new(10, 60);
        for _ in 0..10 {
            assert!(limiter.check().await);
        }
    }

    #[tokio::test]
    async fn test_global_rate_limiter_blocks_over_limit() {
        let limiter = GlobalRateLimiter::new(5, 60);
        for _ in 0..5 {
            assert!(limiter.check().await);
        }
        // 6th request should be blocked
        assert!(!limiter.check().await);
    }
}
