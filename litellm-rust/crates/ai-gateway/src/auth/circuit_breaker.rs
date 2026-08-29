//! Circuit breaker for upstream providers.
//!
//! Prevents cascading failures when providers are down or slow.
//! Uses a simple state machine: Closed -> Open -> HalfOpen -> Closed.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests pass through.
    Closed,
    /// Circuit is tripped - requests fail fast.
    Open,
    /// Testing if provider has recovered - limited requests pass through.
    HalfOpen,
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
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 2,
            max_concurrent: 0,
        }
    }
}

/// Circuit breaker for a single provider.
pub struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: RwLock<u32>,
    success_count: RwLock<u32>,
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
            state: RwLock::new(CircuitState::Closed),
            failure_count: RwLock::new(0),
            success_count: RwLock::new(0),
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
        // Fast path: Closed state is a pure read, no write lock needed.
        // This is the common case (>99% of requests in healthy operation).
        if *self.state.read().await == CircuitState::Closed {
            return true;
        }

        // Slow path: Open or HalfOpen need write lock for state transitions.
        let mut state = self.state.write().await;

        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let last_failure = self.last_failure.read().await;
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.recovery_timeout {
                        *state = CircuitState::HalfOpen;
                        *self.success_count.write().await = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                let success_count = *self.success_count.read().await;
                success_count < self.config.success_threshold
            }
        }
    }

    /// Record a successful request.
    pub async fn record_success(&self) {
        let state = *self.state.read().await;
        
        if state == CircuitState::HalfOpen {
            let mut success_count = self.success_count.write().await;
            *success_count += 1;
            
            if *success_count >= self.config.success_threshold {
                // Transition to closed
                *self.state.write().await = CircuitState::Closed;
                *self.failure_count.write().await = 0;
            }
        } else if state == CircuitState::Closed {
            // Reset failure count on success
            *self.failure_count.write().await = 0;
        }
    }

    /// Record a failed request.
    pub async fn record_failure(&self) {
        let state = *self.state.read().await;
        
        match state {
            CircuitState::Closed => {
                let mut failure_count = self.failure_count.write().await;
                *failure_count += 1;
                *self.last_failure.write().await = Some(Instant::now());
                
                if *failure_count >= self.config.failure_threshold {
                    // Transition to open
                    *self.state.write().await = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open transitions back to open
                *self.state.write().await = CircuitState::Open;
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
        *self.state.read().await
    }

    /// Reset the circuit breaker to closed state.
    pub async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        *self.failure_count.write().await = 0;
        *self.success_count.write().await = 0;
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
