//! Sliding window rate limiter for more accurate rate limiting.
//!
//! Uses a sliding window algorithm instead of fixed windows to provide
//! smoother rate limiting without the boundary issues of fixed windows.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Sliding window rate limiter configuration.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Maximum number of requests allowed in the window.
    pub max_requests: u64,
    /// Size of the sliding window.
    pub window_size: Duration,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            window_size: Duration::from_secs(60),
        }
    }
}

/// A single request timestamp.
#[derive(Debug, Clone, Copy)]
struct RequestTimestamp {
    timestamp: Instant,
}

/// Sliding window rate limiter.
///
/// Tracks request timestamps in a sliding window and rejects requests
/// that would exceed the rate limit.
pub struct SlidingWindowRateLimiter {
    config: SlidingWindowConfig,
    /// Request timestamps per key.
    requests: Arc<Mutex<std::collections::HashMap<String, VecDeque<RequestTimestamp>>>>,
}

impl SlidingWindowRateLimiter {
    /// Create a new sliding window rate limiter.
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self {
            config,
            requests: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Check if a request is allowed and record it if so.
    ///
    /// Returns true if the request is allowed, false if it would exceed the rate limit.
    pub fn check_and_record(&self, key: &str) -> bool {
        let now = Instant::now();
        let window_start = now - self.config.window_size;

        let mut requests = self.requests.lock();
        let timestamps = requests.entry(key.to_string()).or_default();

        // Remove timestamps outside the window
        while let Some(front) = timestamps.front() {
            if front.timestamp < window_start {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        // Check if we're at the limit
        if timestamps.len() as u64 >= self.config.max_requests {
            return false;
        }

        // Record the request
        timestamps.push_back(RequestTimestamp { timestamp: now });
        true
    }

    /// Check if a request is allowed without recording it.
    pub fn check(&self, key: &str) -> bool {
        let now = Instant::now();
        let window_start = now - self.config.window_size;

        let requests = self.requests.lock();
        if let Some(timestamps) = requests.get(key) {
            // Count timestamps within the window
            let count = timestamps
                .iter()
                .filter(|ts| ts.timestamp >= window_start)
                .count();
            (count as u64) < self.config.max_requests
        } else {
            true
        }
    }

    /// Get the current request count for a key.
    pub fn get_count(&self, key: &str) -> u64 {
        let now = Instant::now();
        let window_start = now - self.config.window_size;

        let requests = self.requests.lock();
        if let Some(timestamps) = requests.get(key) {
            timestamps
                .iter()
                .filter(|ts| ts.timestamp >= window_start)
                .count() as u64
        } else {
            0
        }
    }

    /// Reset the rate limiter for a key.
    pub fn reset(&self, key: &str) {
        let mut requests = self.requests.lock();
        requests.remove(key);
    }

    /// Get the number of remaining requests allowed for a key.
    pub fn remaining(&self, key: &str) -> u64 {
        let count = self.get_count(key);
        self.config.max_requests.saturating_sub(count)
    }

    /// Get the time until the next request will be allowed.
    pub fn retry_after(&self, key: &str) -> Option<Duration> {
        let now = Instant::now();
        let window_start = now - self.config.window_size;

        let requests = self.requests.lock();
        if let Some(timestamps) = requests.get(key)
            && timestamps.len() as u64 >= self.config.max_requests
        {
            // Find the oldest timestamp in the window
            if let Some(oldest) = timestamps.iter().find(|ts| ts.timestamp >= window_start) {
                let retry_at = oldest.timestamp + self.config.window_size;
                if retry_at > now {
                    return Some(retry_at - now);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_sliding_window_allows_requests_within_limit() {
        let config = SlidingWindowConfig {
            max_requests: 5,
            window_size: Duration::from_secs(1),
        };
        let limiter = SlidingWindowRateLimiter::new(config);

        for _ in 0..5 {
            assert!(limiter.check_and_record("test_key"));
        }

        assert!(!limiter.check_and_record("test_key"));
    }

    #[test]
    fn test_sliding_window_different_keys() {
        let config = SlidingWindowConfig {
            max_requests: 2,
            window_size: Duration::from_secs(1),
        };
        let limiter = SlidingWindowRateLimiter::new(config);

        assert!(limiter.check_and_record("key1"));
        assert!(limiter.check_and_record("key1"));
        assert!(!limiter.check_and_record("key1"));

        assert!(limiter.check_and_record("key2"));
        assert!(limiter.check_and_record("key2"));
        assert!(!limiter.check_and_record("key2"));
    }

    #[test]
    fn test_sliding_window_expiry() {
        let config = SlidingWindowConfig {
            max_requests: 2,
            window_size: Duration::from_millis(100),
        };
        let limiter = SlidingWindowRateLimiter::new(config);

        assert!(limiter.check_and_record("test_key"));
        assert!(limiter.check_and_record("test_key"));
        assert!(!limiter.check_and_record("test_key"));

        // Wait for window to expire
        thread::sleep(Duration::from_millis(150));

        assert!(limiter.check_and_record("test_key"));
    }

    #[test]
    fn test_sliding_window_reset() {
        let config = SlidingWindowConfig {
            max_requests: 2,
            window_size: Duration::from_secs(1),
        };
        let limiter = SlidingWindowRateLimiter::new(config);

        assert!(limiter.check_and_record("test_key"));
        assert!(limiter.check_and_record("test_key"));
        assert!(!limiter.check_and_record("test_key"));

        limiter.reset("test_key");

        assert!(limiter.check_and_record("test_key"));
    }

    #[test]
    fn test_sliding_window_remaining() {
        let config = SlidingWindowConfig {
            max_requests: 5,
            window_size: Duration::from_secs(1),
        };
        let limiter = SlidingWindowRateLimiter::new(config);

        assert_eq!(limiter.remaining("test_key"), 5);

        limiter.check_and_record("test_key");
        assert_eq!(limiter.remaining("test_key"), 4);

        limiter.check_and_record("test_key");
        assert_eq!(limiter.remaining("test_key"), 3);
    }

    #[test]
    fn test_sliding_window_retry_after() {
        let config = SlidingWindowConfig {
            max_requests: 2,
            window_size: Duration::from_secs(1),
        };
        let limiter = SlidingWindowRateLimiter::new(config);

        assert!(limiter.retry_after("test_key").is_none());

        limiter.check_and_record("test_key");
        limiter.check_and_record("test_key");

        let retry_after = limiter.retry_after("test_key");
        assert!(retry_after.is_some());
        assert!(retry_after.unwrap() <= Duration::from_secs(1));
    }
}
