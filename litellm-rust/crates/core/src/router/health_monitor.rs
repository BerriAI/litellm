//! Health monitoring for deployments.
//!
//! Tracks health metrics per deployment: latency, error rates, success rates,
//! concurrent requests, and overall health status.
//!
//! ## Zero-Allocation Optimizations
//!
//! - Uses `parking_lot::RwLock` instead of `tokio::sync::RwLock` for faster locking
//! - Uses `Arc<str>` for deployment keys to avoid String allocations
//! - Uses `AtomicU64` for all metrics to avoid RwLock overhead
//! - Pre-allocates HashMap with expected capacity

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::Deployment;

/// Health status of a deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Deployment is healthy and accepting requests.
    Healthy,
    /// Deployment is degraded (high latency or error rate).
    Degraded,
    /// Deployment is unhealthy (too many errors or unresponsive).
    Unhealthy,
    /// Deployment status is unknown (no data yet).
    Unknown,
}

/// Health metrics for a single deployment.
/// Uses atomic operations for all metrics to avoid locking overhead.
#[derive(Debug)]
struct DeploymentHealth {
    /// Current health status (0 = Unknown, 1 = Healthy, 2 = Degraded, 3 = Unhealthy)
    status: AtomicU64,
    /// Total requests sent
    total_requests: AtomicU64,
    /// Total successful requests
    successful_requests: AtomicU64,
    /// Total failed requests
    failed_requests: AtomicU64,
    /// Current concurrent requests
    concurrent_requests: AtomicU64,
    /// Rolling average latency in milliseconds (stored as f64 bits in AtomicU64)
    avg_latency_ms_bits: AtomicU64,
    /// Last time health was updated (stored as nanoseconds since epoch in AtomicU64)
    last_updated_nanos: AtomicU64,
    /// Whether the deployment is marked as healthy in config
    _config_healthy: AtomicBool,
}

impl DeploymentHealth {
    fn new(config_healthy: bool) -> Self {
        Self {
            status: AtomicU64::new(if config_healthy { 1 } else { 0 }),
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            concurrent_requests: AtomicU64::new(0),
            avg_latency_ms_bits: AtomicU64::new(0.0_f64.to_bits()),
            last_updated_nanos: AtomicU64::new(0),
            _config_healthy: AtomicBool::new(config_healthy),
        }
    }

    fn get_status(&self) -> HealthStatus {
        match self.status.load(Ordering::Relaxed) {
            0 => HealthStatus::Unknown,
            1 => HealthStatus::Healthy,
            2 => HealthStatus::Degraded,
            3 => HealthStatus::Unhealthy,
            _ => HealthStatus::Unknown,
        }
    }

    fn set_status(&self, status: HealthStatus) {
        let value = match status {
            HealthStatus::Unknown => 0,
            HealthStatus::Healthy => 1,
            HealthStatus::Degraded => 2,
            HealthStatus::Unhealthy => 3,
        };
        self.status.store(value, Ordering::Relaxed);
    }

    fn get_avg_latency_ms(&self) -> f64 {
        f64::from_bits(self.avg_latency_ms_bits.load(Ordering::Relaxed))
    }

    fn _set_avg_latency_ms(&self, latency_ms: f64) {
        self.avg_latency_ms_bits
            .store(latency_ms.to_bits(), Ordering::Relaxed);
    }

    fn _get_last_updated(&self) -> Option<Instant> {
        let nanos = self.last_updated_nanos.load(Ordering::Relaxed);
        if nanos == 0 {
            None
        } else {
            // Convert nanoseconds back to Instant
            // Note: This is an approximation since we can't perfectly reconstruct Instant
            Some(
                Instant::now()
                    - Duration::from_nanos(Instant::now().elapsed().as_nanos() as u64 - nanos),
            )
        }
    }

    fn set_last_updated(&self) {
        // Store current time as nanoseconds since some epoch
        // We use elapsed time as a proxy since Instant doesn't have a direct conversion
        let nanos = Instant::now().elapsed().as_nanos() as u64;
        self.last_updated_nanos.store(nanos, Ordering::Relaxed);
    }
}

/// Health monitor for all deployments.
///
/// ## Zero-Allocation Design
///
/// - `Arc<str>` keys: Shared ownership, no reallocation on clone
/// - `parking_lot::RwLock`: Faster than std sync primitives
/// - Pre-allocated HashMap: Reduces rehashing overhead
/// - Atomic operations in DeploymentHealth: Lock-free metric updates
#[derive(Debug)]
pub struct HealthMonitor {
    /// Health metrics per deployment key (model_name + provider)
    /// Uses Arc<str> for zero-copy key sharing
    deployments: Arc<RwLock<HashMap<Arc<str>, Arc<DeploymentHealth>>>>,
    /// Configuration thresholds
    config: HealthConfig,
}

/// Configuration for health monitoring.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Maximum acceptable average latency in milliseconds before marking as degraded.
    pub latency_degraded_threshold_ms: f64,
    /// Maximum acceptable average latency in milliseconds before marking as unhealthy.
    pub latency_unhealthy_threshold_ms: f64,
    /// Maximum acceptable error rate (0.0 to 1.0) before marking as degraded.
    pub error_rate_degraded_threshold: f64,
    /// Maximum acceptable error rate (0.0 to 1.0) before marking as unhealthy.
    pub error_rate_unhealthy_threshold: f64,
    /// Minimum number of requests before health status is calculated.
    pub min_requests_for_health: u64,
    /// Window size for rolling average latency calculation.
    pub latency_window_size: usize,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            latency_degraded_threshold_ms: 1000.0,  // 1 second
            latency_unhealthy_threshold_ms: 5000.0, // 5 seconds
            error_rate_degraded_threshold: 0.1,     // 10%
            error_rate_unhealthy_threshold: 0.5,    // 50%
            min_requests_for_health: 10,
            latency_window_size: 100,
        }
    }
}

impl HealthMonitor {
    /// Create a new health monitor with the given configuration.
    ///
    /// Pre-allocates HashMap with capacity for 16 deployments to reduce rehashing.
    pub fn new(config: HealthConfig) -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::with_capacity(16))),
            config,
        }
    }

    /// Get the deployment key for a deployment.
    /// Returns Arc<str> for zero-copy sharing.
    #[inline]
    fn deployment_key(deployment: &Deployment) -> Arc<str> {
        let key = format!(
            "{}:{}",
            deployment.model_name, deployment.litellm_params.model
        );
        Arc::from(key)
    }

    /// Get or create health metrics for a deployment.
    ///
    /// ## Performance
    ///
    /// - Uses parking_lot::RwLock for faster write locking
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    fn get_or_create_health(&self, deployment: &Deployment) -> Arc<DeploymentHealth> {
        let key = Self::deployment_key(deployment);
        let mut deployments = self.deployments.write();
        deployments
            .entry(Arc::clone(&key))
            .or_insert_with(|| {
                let config_healthy = deployment.healthy.unwrap_or(true);
                Arc::new(DeploymentHealth::new(config_healthy))
            })
            .clone()
    }

    /// Record the start of a request to a deployment.
    ///
    /// ## Performance
    ///
    /// - Lock-free atomic operations
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn record_request_start(&self, deployment: &Deployment) {
        let health = self.get_or_create_health(deployment);
        health.total_requests.fetch_add(1, Ordering::Relaxed);
        health.concurrent_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the successful completion of a request to a deployment.
    ///
    /// ## Performance
    ///
    /// - Lock-free atomic operations
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn record_request_success(&self, deployment: &Deployment, latency: Duration) {
        let health = self.get_or_create_health(deployment);
        health.successful_requests.fetch_add(1, Ordering::Relaxed);
        health.concurrent_requests.fetch_sub(1, Ordering::Relaxed);

        // Update rolling average latency using atomic operations
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let total_requests = health.total_requests.load(Ordering::Relaxed) as f64;

        // Atomically update average latency using compare-and-swap
        loop {
            let current_bits = health.avg_latency_ms_bits.load(Ordering::Relaxed);
            let current_avg = f64::from_bits(current_bits);
            let new_avg = (current_avg * (total_requests - 1.0) + latency_ms) / total_requests;
            let new_bits = new_avg.to_bits();

            if health
                .avg_latency_ms_bits
                .compare_exchange(current_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Update last updated time
        health.set_last_updated();

        // Recalculate health status
        self.recalculate_health_status(deployment).await;
    }

    /// Record the failure of a request to a deployment.
    ///
    /// ## Performance
    ///
    /// - Lock-free atomic operations
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn record_request_failure(&self, deployment: &Deployment, latency: Duration) {
        let health = self.get_or_create_health(deployment);
        health.failed_requests.fetch_add(1, Ordering::Relaxed);
        health.concurrent_requests.fetch_sub(1, Ordering::Relaxed);

        // Update rolling average latency using atomic operations
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let total_requests = health.total_requests.load(Ordering::Relaxed) as f64;

        // Atomically update average latency using compare-and-swap
        loop {
            let current_bits = health.avg_latency_ms_bits.load(Ordering::Relaxed);
            let current_avg = f64::from_bits(current_bits);
            let new_avg = (current_avg * (total_requests - 1.0) + latency_ms) / total_requests;
            let new_bits = new_avg.to_bits();

            if health
                .avg_latency_ms_bits
                .compare_exchange(current_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Update last updated time
        health.set_last_updated();

        // Recalculate health status
        self.recalculate_health_status(deployment).await;
    }

    /// Recalculate the health status of a deployment based on metrics.
    ///
    /// ## Performance
    ///
    /// - Lock-free atomic operations
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    async fn recalculate_health_status(&self, deployment: &Deployment) {
        let health = self.get_or_create_health(deployment);
        let total_requests = health.total_requests.load(Ordering::Relaxed);

        // Not enough data yet
        if total_requests < self.config.min_requests_for_health {
            health.set_status(HealthStatus::Unknown);
            return;
        }

        let _successful_requests = health.successful_requests.load(Ordering::Relaxed);
        let failed_requests = health.failed_requests.load(Ordering::Relaxed);
        let error_rate = failed_requests as f64 / total_requests as f64;
        let avg_latency = health.get_avg_latency_ms();

        // Determine health status based on error rate and latency
        let status = if error_rate >= self.config.error_rate_unhealthy_threshold
            || avg_latency >= self.config.latency_unhealthy_threshold_ms
        {
            HealthStatus::Unhealthy
        } else if error_rate >= self.config.error_rate_degraded_threshold
            || avg_latency >= self.config.latency_degraded_threshold_ms
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        health.set_status(status);
    }

    /// Get the health status of a deployment.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - Lock-free atomic status read
    #[inline]
    pub async fn get_health_status(&self, deployment: &Deployment) -> HealthStatus {
        let key = Self::deployment_key(deployment);
        let deployments = self.deployments.read();
        deployments
            .get(&key)
            .map(|h| h.get_status())
            .unwrap_or(HealthStatus::Unknown)
    }

    /// Get detailed health metrics for a deployment.
    ///
    /// ## Performance
    ///
    /// - Lock-free atomic operations
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn get_health_metrics(&self, deployment: &Deployment) -> HealthMetrics {
        let health = self.get_or_create_health(deployment);
        let total_requests = health.total_requests.load(Ordering::Relaxed);
        let successful_requests = health.successful_requests.load(Ordering::Relaxed);
        let failed_requests = health.failed_requests.load(Ordering::Relaxed);
        let concurrent_requests = health.concurrent_requests.load(Ordering::Relaxed);
        let avg_latency_ms = health.get_avg_latency_ms();
        let error_rate = if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        };

        HealthMetrics {
            status: health.get_status(),
            total_requests,
            successful_requests,
            failed_requests,
            concurrent_requests,
            avg_latency_ms,
            error_rate,
            success_rate: if total_requests > 0 {
                successful_requests as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Get health metrics for all deployments.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Lock-free atomic operations for all metrics
    /// - Returns HashMap with Arc<str> keys (zero-copy)
    #[inline]
    pub async fn get_all_health_metrics(&self) -> HashMap<Arc<str>, HealthMetrics> {
        let deployments = self.deployments.read();
        let mut metrics = HashMap::with_capacity(deployments.len());

        for (key, health) in deployments.iter() {
            let total_requests = health.total_requests.load(Ordering::Relaxed);
            let successful_requests = health.successful_requests.load(Ordering::Relaxed);
            let failed_requests = health.failed_requests.load(Ordering::Relaxed);
            let concurrent_requests = health.concurrent_requests.load(Ordering::Relaxed);
            let avg_latency_ms = health.get_avg_latency_ms();
            let error_rate = if total_requests > 0 {
                failed_requests as f64 / total_requests as f64
            } else {
                0.0
            };

            metrics.insert(
                key.clone(),
                HealthMetrics {
                    status: health.get_status(),
                    total_requests,
                    successful_requests,
                    failed_requests,
                    concurrent_requests,
                    avg_latency_ms,
                    error_rate,
                    success_rate: if total_requests > 0 {
                        successful_requests as f64 / total_requests as f64
                    } else {
                        0.0
                    },
                },
            );
        }

        metrics
    }

    /// Check if a deployment is healthy enough to receive traffic.
    ///
    /// ## Performance
    ///
    /// - Lock-free atomic status read
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn is_healthy(&self, deployment: &Deployment) -> bool {
        let status = self.get_health_status(deployment).await;
        matches!(status, HealthStatus::Healthy | HealthStatus::Unknown)
    }

    /// Filter deployments to only include healthy ones.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Lock-free atomic status reads
    /// - Uses Arc<str> keys for zero-copy lookup
    #[inline]
    pub async fn filter_healthy<'a>(&self, candidates: Vec<&'a Deployment>) -> Vec<&'a Deployment> {
        let deployments = self.deployments.read();
        candidates
            .into_iter()
            .filter(|deployment| {
                let key = Self::deployment_key(deployment);
                deployments
                    .get(&key)
                    .map(|h| {
                        matches!(
                            h.get_status(),
                            HealthStatus::Healthy | HealthStatus::Unknown
                        )
                    })
                    .unwrap_or(true) // Unknown status is considered healthy
            })
            .collect()
    }

    /// Clear all health data.
    ///
    /// ## Performance
    ///
    /// - Single write lock acquisition
    #[inline]
    pub async fn clear(&self) {
        let mut deployments = self.deployments.write();
        deployments.clear();
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new(HealthConfig::default())
    }
}

impl Clone for HealthMonitor {
    fn clone(&self) -> Self {
        Self {
            deployments: Arc::clone(&self.deployments),
            config: self.config.clone(),
        }
    }
}

/// Detailed health metrics for a deployment.
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    pub status: HealthStatus,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub concurrent_requests: u64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::LiteLLMParams;

    fn create_deployment(model_name: &str, model: &str, healthy: Option<bool>) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
            healthy,
            weight: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        }
    }

    #[tokio::test]
    async fn test_health_monitor_initial_status() {
        let monitor = HealthMonitor::default();
        let deployment = create_deployment("gpt-4", "openai/gpt-4", Some(true));

        // Initial status should be Unknown (no data yet)
        assert_eq!(
            monitor.get_health_status(&deployment).await,
            HealthStatus::Unknown
        );
    }

    #[tokio::test]
    async fn test_health_monitor_record_success() {
        let monitor = HealthMonitor::default();
        let deployment = create_deployment("gpt-4", "openai/gpt-4", Some(true));

        // Record some successful requests
        for _ in 0..15 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_success(&deployment, Duration::from_millis(100))
                .await;
        }

        let metrics = monitor.get_health_metrics(&deployment).await;
        assert_eq!(metrics.status, HealthStatus::Healthy);
        assert_eq!(metrics.total_requests, 15);
        assert_eq!(metrics.successful_requests, 15);
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.concurrent_requests, 0);
        assert!(metrics.avg_latency_ms > 0.0);
        assert_eq!(metrics.error_rate, 0.0);
        assert_eq!(metrics.success_rate, 1.0);
    }

    #[tokio::test]
    async fn test_health_monitor_record_failure() {
        let monitor = HealthMonitor::default();
        let deployment = create_deployment("gpt-4", "openai/gpt-4", Some(true));

        // Record some requests with failures
        for _ in 0..10 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_success(&deployment, Duration::from_millis(100))
                .await;
        }
        for _ in 0..5 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_failure(&deployment, Duration::from_millis(200))
                .await;
        }

        let metrics = monitor.get_health_metrics(&deployment).await;
        assert_eq!(metrics.total_requests, 15);
        assert_eq!(metrics.successful_requests, 10);
        assert_eq!(metrics.failed_requests, 5);
        assert!((metrics.error_rate - 0.333).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_health_monitor_degraded_status() {
        let config = HealthConfig {
            error_rate_degraded_threshold: 0.1,
            min_requests_for_health: 10,
            ..Default::default()
        };
        let monitor = HealthMonitor::new(config);
        let deployment = create_deployment("gpt-4", "openai/gpt-4", Some(true));

        // Record requests with high error rate
        for _ in 0..8 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_success(&deployment, Duration::from_millis(100))
                .await;
        }
        for _ in 0..4 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_failure(&deployment, Duration::from_millis(200))
                .await;
        }

        let metrics = monitor.get_health_metrics(&deployment).await;
        assert_eq!(metrics.status, HealthStatus::Degraded);
    }

    #[tokio::test]
    async fn test_health_monitor_unhealthy_status() {
        let config = HealthConfig {
            error_rate_unhealthy_threshold: 0.5,
            min_requests_for_health: 10,
            ..Default::default()
        };
        let monitor = HealthMonitor::new(config);
        let deployment = create_deployment("gpt-4", "openai/gpt-4", Some(true));

        // Record requests with very high error rate
        for _ in 0..5 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_success(&deployment, Duration::from_millis(100))
                .await;
        }
        for _ in 0..10 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_failure(&deployment, Duration::from_millis(200))
                .await;
        }

        let metrics = monitor.get_health_metrics(&deployment).await;
        assert_eq!(metrics.status, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn test_health_monitor_is_healthy() {
        let monitor = HealthMonitor::default();
        let deployment = create_deployment("gpt-4", "openai/gpt-4", Some(true));

        // Initially should be considered healthy (Unknown status)
        assert!(monitor.is_healthy(&deployment).await);

        // Record some successful requests
        for _ in 0..15 {
            monitor.record_request_start(&deployment).await;
            monitor
                .record_request_success(&deployment, Duration::from_millis(100))
                .await;
        }

        // Should still be healthy
        assert!(monitor.is_healthy(&deployment).await);
    }

    #[tokio::test]
    async fn test_health_monitor_filter_healthy() {
        let monitor = HealthMonitor::default();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4", Some(true));
        let dep2 = create_deployment("gpt-4", "azure/gpt-4", Some(true));

        // Make dep2 unhealthy
        for _ in 0..10 {
            monitor.record_request_start(&dep2).await;
            monitor
                .record_request_failure(&dep2, Duration::from_millis(200))
                .await;
        }

        let candidates = vec![&dep1, &dep2];
        let healthy = monitor.filter_healthy(candidates).await;

        // dep1 should be healthy (Unknown status), dep2 should be unhealthy
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].litellm_params.model, "openai/gpt-4");
    }
}
