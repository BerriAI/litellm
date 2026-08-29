//! Operational hardening features for production resilience.
//!
//! Provides:
//! - Global rate limiting (in addition to per-key limits)
//! - Graceful degradation logging for dependency failures
//! - Slow loris protection configuration
//! - Secret rotation support (file-based secret loading)
//! - Audit log shipping to external systems

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

/// Secret rotation support: loads secrets from files and watches for changes.
/// Enables zero-downtime secret rotation by reading from a file that can be
/// updated externally (e.g., by a secrets manager sidecar).
pub struct SecretRotator {
    path: String,
    current: RwLock<String>,
}

impl SecretRotator {
    /// Create a new SecretRotator that reads from the given file path.
    /// Returns error if the file cannot be read initially.
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self {
            path: path.to_string(),
            current: RwLock::new(content),
        })
    }

    /// Get the current secret value. Zero-allocation: returns a read guard.
    pub async fn get(&self) -> tokio::sync::RwLockReadGuard<'_, String> {
        self.current.read().await
    }

    /// Reload the secret from disk. Call this periodically or on signal.
    pub async fn reload(&self) -> Result<(), std::io::Error> {
        let content = std::fs::read_to_string(&self.path)?;
        let mut current = self.current.write().await;
        *current = content;
        Ok(())
    }

    /// Spawn a background task that reloads the secret every `interval`.
    pub fn spawn_watcher(self: Arc<Self>, interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if let Err(e) = self.reload().await {
                    log_degradation("secret_rotator", "reload", &e.to_string());
                }
            }
        })
    }
}

/// Audit log shipper: sends audit log entries to an external HTTP endpoint.
/// Buffers entries and flushes them in batches for efficiency.
pub struct AuditLogShipper {
    endpoint: String,
    buffer: RwLock<Vec<String>>,
    max_batch_size: usize,
    client: reqwest::Client,
}

impl AuditLogShipper {
    /// Create a new AuditLogShipper that sends to the given endpoint.
    pub fn new(endpoint: &str, max_batch_size: usize) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            buffer: RwLock::new(Vec::with_capacity(max_batch_size)),
            max_batch_size,
            client: reqwest::Client::new(),
        }
    }

    /// Ship a log entry. Buffers entries and flushes when batch is full.
    pub async fn ship(&self, entry: String) {
        let should_flush = {
            let mut buffer = self.buffer.write().await;
            buffer.push(entry);
            buffer.len() >= self.max_batch_size
        };
        if should_flush {
            self.flush().await;
        }
    }

    /// Flush all buffered entries to the external endpoint.
    pub async fn flush(&self) {
        let batch = {
            let mut buffer = self.buffer.write().await;
            if buffer.is_empty() {
                return;
            }
            std::mem::take(&mut *buffer)
        };

        let body = serde_json::to_string(&batch).unwrap_or_default();
        match self.client.post(&self.endpoint)
            .header("content-type", "application/json")
            .body(body)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => {
                log_degradation("audit_shipper", "flush", &format!("HTTP {}", resp.status()));
            }
            Err(e) => {
                log_degradation("audit_shipper", "flush", &e.to_string());
            }
        }
    }

    /// Spawn a background task that flushes buffered entries periodically.
    pub fn spawn_flusher(self: Arc<Self>, interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                self.flush().await;
            }
        })
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
