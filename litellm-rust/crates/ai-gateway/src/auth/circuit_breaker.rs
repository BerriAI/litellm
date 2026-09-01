//! Circuit breaker for upstream providers.
//!
//! Prevents cascading failures when providers are down or slow.
//! Uses a simple state machine: Closed -> Open -> HalfOpen -> Closed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitState {
    /// Normal operation - requests pass through.
    Closed = 0,
    /// Circuit is tripped - requests fail fast.
    Open = 1,
    /// Testing if provider has recovered - limited requests pass through.
    HalfOpen = 2,
}

impl From<u8> for CircuitState {
    fn from(val: u8) -> Self {
        match val {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit.
    pub failure_threshold: u32,
    /// Time to wait before trying again after opening.
    pub recovery_timeout: Duration,
    /// Number of successful requests in half-open state to close the circuit.
    pub success_threshold: u32,
    /// Maximum concurrent requests per provider (bulkhead). 0 means unlimited.
    pub max_concurrent: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 50,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 2,
            max_concurrent: 0,
        }
    }
}

/// Circuit breaker for a single provider.
pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure: RwLock<Option<Instant>>,
    config: CircuitBreakerConfig,
    semaphore: Option<Arc<Semaphore>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let semaphore = if config.max_concurrent > 0 {
            Some(Arc::new(Semaphore::new(config.max_concurrent)))
        } else {
            None
        };
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure: RwLock::new(None),
            config,
            semaphore,
        }
    }

    /// Try to acquire a bulkhead slot. Returns None if bulkhead is full or not configured.
    pub async fn try_acquire(&self) -> Option<tokio::sync::SemaphorePermit<'_>> {
        match &self.semaphore {
            Some(sem) => sem.try_acquire().ok(),
            None => None,
        }
    }

    /// Check if a request should be allowed through.
    pub async fn allow_request(&self) -> bool {
        // Fast path: Closed state is a pure atomic read, no lock needed.
        // This is the common case (>99% of requests in healthy operation).
        let state = CircuitState::from(self.state.load(Ordering::Relaxed));
        if state == CircuitState::Closed {
            return true;
        }

        // Slow path: Open or HalfOpen need to check transitions.
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let last_failure = self.last_failure.read().await;
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.recovery_timeout {
                        // Transition to HalfOpen
                        self.state
                            .store(CircuitState::HalfOpen as u8, Ordering::Relaxed);
                        self.success_count.store(0, Ordering::Relaxed);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.load(Ordering::Relaxed);
                success_count < self.config.success_threshold
            }
        }
    }

    /// Record a successful request.
    pub async fn record_success(&self) {
        let state = CircuitState::from(self.state.load(Ordering::Relaxed));

        if state == CircuitState::HalfOpen {
            let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;

            if success_count >= self.config.success_threshold {
                // Transition to closed
                self.state
                    .store(CircuitState::Closed as u8, Ordering::Relaxed);
                self.failure_count.store(0, Ordering::Relaxed);
            }
        } else if state == CircuitState::Closed {
            // Reset failure count on success
            self.failure_count.store(0, Ordering::Relaxed);
        }
    }

    /// Record a failed request.
    pub async fn record_failure(&self) {
        let state = CircuitState::from(self.state.load(Ordering::Relaxed));

        match state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                *self.last_failure.write().await = Some(Instant::now());

                if failure_count >= self.config.failure_threshold {
                    // Transition to open
                    self.state
                        .store(CircuitState::Open as u8, Ordering::Relaxed);
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open transitions back to open
                self.state
                    .store(CircuitState::Open as u8, Ordering::Relaxed);
                *self.last_failure.write().await = Some(Instant::now());
            }
            CircuitState::Open => {
                // Already open, just update last failure time
                *self.last_failure.write().await = Some(Instant::now());
            }
        }
    }

    /// Get the current state of the circuit breaker.
    pub async fn state(&self) -> CircuitState {
        CircuitState::from(self.state.load(Ordering::Relaxed))
    }

    /// Reset the circuit breaker to closed state.
    pub async fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        *self.last_failure.write().await = None;
    }
}

/// Registry of circuit breakers for multiple providers.
pub struct CircuitBreakerRegistry {
    breakers: std::sync::RwLock<std::collections::HashMap<String, Arc<CircuitBreaker>>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: std::sync::RwLock::new(std::collections::HashMap::new()),
            config,
        }
    }

    /// Get or create a circuit breaker for a provider.
    pub fn get_or_create(&self, provider: &str) -> Arc<CircuitBreaker> {
        // Fast path: read lock, breaker already exists.
        {
            let breakers = self.breakers.read().unwrap();
            if let Some(breaker) = breakers.get(provider) {
                return breaker.clone();
            }
        }

        // Slow path: write lock, create the breaker.
        let mut breakers = self.breakers.write().unwrap();
        breakers
            .entry(provider.to_string())
            .or_insert_with(|| Arc::new(CircuitBreaker::new(self.config.clone())))
            .clone()
    }

    /// Get a circuit breaker for a provider if it exists.
    pub fn get(&self, provider: &str) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.breakers.read().unwrap();
        breakers.get(provider).cloned()
    }
}
