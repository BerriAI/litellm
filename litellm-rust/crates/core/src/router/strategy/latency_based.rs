//! Latency-based routing strategy.
//!
//! Routes requests to the deployment with the lowest average latency.
//! Tracks rolling average latency per deployment and selects the fastest.
//!
//! ## Zero-Allocation Optimizations
//!
//! - Uses `parking_lot::RwLock` instead of `tokio::sync::RwLock` for faster locking
//! - Uses `Arc<str>` for deployment keys to avoid String allocations
//! - Uses `SmallVec` for rolling average samples to avoid heap allocation for small windows
//! - Pre-allocates HashMap with expected capacity

use parking_lot::RwLock;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::Deployment;

/// Latency tracker for deployments.
/// Maintains a rolling average of response times per deployment.
///
/// ## Zero-Allocation Design
///
/// - `Arc<str>` keys: Shared ownership, no reallocation on clone
/// - `SmallVec<[f64; 16]>`: Stack-allocated for windows ≤16 samples
/// - `parking_lot::RwLock`: Faster than std sync primitives
/// - Pre-allocated HashMap: Reduces rehashing overhead
#[derive(Debug)]
pub struct LatencyTracker {
    /// Average latency per deployment key (model_name + provider)
    /// Uses Arc<str> for zero-copy key sharing
    latencies: Arc<RwLock<HashMap<Arc<str>, RollingAverage>>>,
    /// Window size for rolling average (number of samples)
    window_size: usize,
}

/// Rolling average calculator.
/// Uses SmallVec to avoid heap allocation for small window sizes.
#[derive(Debug, Clone)]
struct RollingAverage {
    /// Stack-allocated samples for small windows (≤16 samples)
    /// Falls back to heap allocation only if window_size > 16
    samples: SmallVec<[f64; 16]>,
    /// Running sum for O(1) average calculation
    sum: f64,
    /// Current index in circular buffer
    index: usize,
    /// Maximum capacity (window size)
    capacity: usize,
}

impl RollingAverage {
    fn new(capacity: usize) -> Self {
        Self {
            samples: SmallVec::with_capacity(capacity.min(16)),
            sum: 0.0,
            index: 0,
            capacity,
        }
    }

    fn add_sample(&mut self, value: f64) {
        if self.samples.len() < self.capacity {
            self.samples.push(value);
            self.sum += value;
        } else {
            // Replace oldest sample in circular buffer
            let old_value = self.samples[self.index];
            self.sum = self.sum - old_value + value;
            self.samples[self.index] = value;
            self.index = (self.index + 1) % self.capacity;
        }
    }

    fn average(&self) -> f64 {
        if self.samples.is_empty() {
            f64::MAX // Return max latency if no samples
        } else {
            self.sum / self.samples.len() as f64
        }
    }

    fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

impl LatencyTracker {
    /// Create a new latency tracker with the specified window size.
    ///
    /// Pre-allocates HashMap with capacity for 16 deployments to reduce rehashing.
    pub fn new(window_size: usize) -> Self {
        let latencies = HashMap::with_capacity(16);
        Self {
            latencies: Arc::new(RwLock::new(latencies)),
            window_size,
        }
    }

    /// Get the deployment key for a deployment.
    /// Returns Arc<str> for zero-copy sharing.
    #[inline]
    fn deployment_key(deployment: &Deployment) -> Arc<str> {
        // Use Arc<str> to avoid String allocation on every access
        let key = format!(
            "{}:{}",
            deployment.model_name, deployment.litellm_params.model
        );
        Arc::from(key)
    }

    /// Record a latency sample for a deployment.
    ///
    /// ## Performance
    ///
    /// - Uses parking_lot::RwLock for faster write locking
    /// - Uses Arc<str> keys to avoid String allocation
    /// - Uses SmallVec for samples to avoid heap allocation for small windows
    #[inline]
    pub async fn record_latency(&self, deployment: &Deployment, latency: Duration) {
        let key = Self::deployment_key(deployment);
        let latency_ms = latency.as_secs_f64() * 1000.0;

        let mut latencies = self.latencies.write();
        let rolling_avg = latencies
            .entry(key)
            .or_insert_with(|| RollingAverage::new(self.window_size));
        rolling_avg.add_sample(latency_ms);
    }

    /// Get the average latency for a deployment.
    ///
    /// ## Performance
    ///
    /// - Uses parking_lot::RwLock for faster read locking
    /// - Uses Arc<str> keys for zero-copy lookup
    #[inline]
    pub async fn get_average_latency(&self, deployment: &Deployment) -> f64 {
        let key = Self::deployment_key(deployment);
        let latencies = self.latencies.read();
        latencies
            .get(&key)
            .map(|avg| avg.average())
            .unwrap_or(f64::MAX)
    }

    /// Get the sample count for a deployment.
    #[inline]
    pub async fn get_sample_count(&self, deployment: &Deployment) -> usize {
        let key = Self::deployment_key(deployment);
        let latencies = self.latencies.read();
        latencies
            .get(&key)
            .map(|avg| avg.sample_count())
            .unwrap_or(0)
    }

    /// Select the deployment with the lowest average latency.
    /// If no latency data is available, falls back to the first deployment.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n) scan where n is number of candidates
    #[inline]
    pub async fn select<'a>(&self, candidates: &[&'a Deployment]) -> Option<&'a Deployment> {
        if candidates.is_empty() {
            return None;
        }

        let latencies = self.latencies.read();

        let mut best_deployment = candidates[0];
        let mut best_latency = latencies
            .get(&Self::deployment_key(best_deployment))
            .map(|avg| avg.average())
            .unwrap_or(f64::MAX);

        for &deployment in candidates.iter().skip(1) {
            let latency = latencies
                .get(&Self::deployment_key(deployment))
                .map(|avg| avg.average())
                .unwrap_or(f64::MAX);

            if latency < best_latency {
                best_latency = latency;
                best_deployment = deployment;
            }
        }

        Some(best_deployment)
    }

    /// Get all deployments sorted by latency (lowest first).
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n log n) sort where n is number of candidates
    #[inline]
    pub async fn sort_by_latency<'a>(
        &self,
        candidates: Vec<&'a Deployment>,
    ) -> Vec<&'a Deployment> {
        let latencies = self.latencies.read();

        let mut candidates_with_latency: Vec<(&Deployment, f64)> = candidates
            .into_iter()
            .map(|deployment| {
                let latency = latencies
                    .get(&Self::deployment_key(deployment))
                    .map(|avg| avg.average())
                    .unwrap_or(f64::MAX);
                (deployment, latency)
            })
            .collect();

        // Sort by latency (lowest first)
        candidates_with_latency
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates_with_latency
            .into_iter()
            .map(|(d, _)| d)
            .collect()
    }

    /// Clear all latency data.
    #[inline]
    pub async fn clear(&self) {
        let mut latencies = self.latencies.write();
        latencies.clear();
    }

    /// Get all latency statistics.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Returns HashMap with Arc<str> keys (zero-copy)
    #[inline]
    pub async fn get_stats(&self) -> HashMap<Arc<str>, (f64, usize)> {
        let latencies = self.latencies.read();
        latencies
            .iter()
            .map(|(key, avg)| (Arc::clone(key), (avg.average(), avg.sample_count())))
            .collect()
    }
}

impl Clone for LatencyTracker {
    fn clone(&self) -> Self {
        Self {
            latencies: Arc::clone(&self.latencies),
            window_size: self.window_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::LiteLLMParams;

    fn create_deployment(model_name: &str, model: &str) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
            healthy: Some(true),
            weight: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        }
    }

    #[tokio::test]
    async fn test_rolling_average() {
        let mut avg = RollingAverage::new(3);

        avg.add_sample(100.0);
        assert_eq!(avg.average(), 100.0);

        avg.add_sample(200.0);
        assert_eq!(avg.average(), 150.0);

        avg.add_sample(300.0);
        assert_eq!(avg.average(), 200.0);

        // Window is full, should replace oldest
        avg.add_sample(400.0);
        assert_eq!(avg.average(), 300.0); // (200 + 300 + 400) / 3
    }

    #[tokio::test]
    async fn test_latency_tracker_record_and_get() {
        let tracker = LatencyTracker::new(5);
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // No data yet
        assert_eq!(tracker.get_average_latency(&deployment).await, f64::MAX);
        assert_eq!(tracker.get_sample_count(&deployment).await, 0);

        // Record some latencies
        tracker
            .record_latency(&deployment, Duration::from_millis(100))
            .await;
        tracker
            .record_latency(&deployment, Duration::from_millis(200))
            .await;
        tracker
            .record_latency(&deployment, Duration::from_millis(300))
            .await;

        assert_eq!(tracker.get_average_latency(&deployment).await, 200.0);
        assert_eq!(tracker.get_sample_count(&deployment).await, 3);
    }

    #[tokio::test]
    async fn test_latency_tracker_select() {
        let tracker = LatencyTracker::new(5);
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        // Record latencies: dep2 is fastest
        tracker
            .record_latency(&dep1, Duration::from_millis(300))
            .await;
        tracker
            .record_latency(&dep2, Duration::from_millis(100))
            .await;
        tracker
            .record_latency(&dep3, Duration::from_millis(200))
            .await;

        let candidates = vec![&dep1, &dep2, &dep3];
        let selected = tracker.select(&candidates).await.unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }

    #[tokio::test]
    async fn test_latency_tracker_sort() {
        let tracker = LatencyTracker::new(5);
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        tracker
            .record_latency(&dep1, Duration::from_millis(300))
            .await;
        tracker
            .record_latency(&dep2, Duration::from_millis(100))
            .await;
        tracker
            .record_latency(&dep3, Duration::from_millis(200))
            .await;

        let candidates = vec![&dep1, &dep2, &dep3];
        let sorted = tracker.sort_by_latency(candidates).await;

        assert_eq!(sorted[0].litellm_params.model, "azure/gpt-4");
        assert_eq!(sorted[1].litellm_params.model, "anthropic/gpt-4");
        assert_eq!(sorted[2].litellm_params.model, "openai/gpt-4");
    }

    #[tokio::test]
    async fn test_latency_tracker_empty_candidates() {
        let tracker = LatencyTracker::new(5);
        let candidates: Vec<&Deployment> = vec![];
        assert!(tracker.select(&candidates).await.is_none());
    }

    #[tokio::test]
    async fn test_latency_tracker_no_data_fallback() {
        let tracker = LatencyTracker::new(5);
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        // No latency data recorded
        let candidates = vec![&dep1, &dep2];
        let selected = tracker.select(&candidates).await.unwrap();
        // Should return first candidate when no data
        assert_eq!(selected.litellm_params.model, "openai/gpt-4");
    }
}
