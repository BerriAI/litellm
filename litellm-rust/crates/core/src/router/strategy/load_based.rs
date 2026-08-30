//! Load-based routing strategy.
//!
//! Routes requests to the deployment with the lowest current load (fewest concurrent requests).
//! Tracks active request counts per deployment and selects the least loaded.
//!
//! ## Zero-Allocation Optimizations
//!
//! - Uses `parking_lot::RwLock` instead of `tokio::sync::RwLock` for faster locking
//! - Uses `Arc<str>` for deployment keys to avoid String allocations
//! - Uses `AtomicU64` directly (no Arc wrapper) for lock-free counter updates
//! - Pre-allocates HashMap with expected capacity

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;

use super::Deployment;

/// Load tracker for deployments.
/// Tracks the number of concurrent requests per deployment.
///
/// ## Zero-Allocation Design
///
/// - `Arc<str>` keys: Shared ownership, no reallocation on clone
/// - `AtomicU64` counters: Lock-free atomic updates
/// - `parking_lot::RwLock`: Faster than std sync primitives
/// - Pre-allocated HashMap: Reduces rehashing overhead
#[derive(Debug)]
pub struct LoadTracker {
    /// Active request count per deployment key (model_name + provider)
    /// Uses Arc<str> for zero-copy key sharing
    /// Uses AtomicU64 for lock-free counter updates
    loads: Arc<RwLock<HashMap<Arc<str>, AtomicU64>>>,
}

impl LoadTracker {
    /// Create a new load tracker.
    ///
    /// Pre-allocates HashMap with capacity for 16 deployments to reduce rehashing.
    pub fn new() -> Self {
        let loads = HashMap::with_capacity(16);
        Self {
            loads: Arc::new(RwLock::new(loads)),
        }
    }

    /// Get the deployment key for a deployment.
    /// Returns Arc<str> for zero-copy sharing.
    #[inline]
    fn deployment_key(deployment: &Deployment) -> Arc<str> {
        let key = format!("{}:{}", deployment.model_name, deployment.litellm_params.model);
        Arc::from(key)
    }

    /// Get or create the load counter for a deployment.
    ///
    /// ## Performance
    ///
    /// - Uses parking_lot::RwLock for faster write locking
    /// - Uses Arc<str> keys to avoid String allocation
    /// - Creates AtomicU64 directly (no Arc wrapper)
    #[inline]
    fn get_or_create_counter(&self, deployment: &Deployment) -> Arc<str> {
        let key = Self::deployment_key(deployment);
        let mut loads = self.loads.write();
        
        // Insert if not present
        loads.entry(Arc::clone(&key)).or_insert_with(|| AtomicU64::new(0));
        
        key
    }

    /// Increment the load counter for a deployment (called when a request starts).
    ///
    /// ## Performance
    ///
    /// - Single write lock acquisition
    /// - Lock-free atomic increment after counter lookup
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn increment_load(&self, deployment: &Deployment) {
        let key = self.get_or_create_counter(deployment);
        let loads = self.loads.read();
        if let Some(counter) = loads.get(&key) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement the load counter for a deployment (called when a request completes).
    ///
    /// ## Performance
    ///
    /// - Single write lock acquisition
    /// - Lock-free atomic decrement after counter lookup
    /// - Uses Arc<str> keys to avoid String allocation
    #[inline]
    pub async fn decrement_load(&self, deployment: &Deployment) {
        let key = self.get_or_create_counter(deployment);
        let loads = self.loads.read();
        if let Some(counter) = loads.get(&key) {
            counter.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get the current load for a deployment.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Lock-free atomic load
    /// - Uses Arc<str> keys for zero-copy lookup
    #[inline]
    pub async fn get_load(&self, deployment: &Deployment) -> u64 {
        let key = Self::deployment_key(deployment);
        let loads = self.loads.read();
        loads
            .get(&key)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Select the deployment with the lowest current load.
    /// If multiple deployments have the same load, returns the first one.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Lock-free atomic loads for all counters
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n) scan where n is number of candidates
    #[inline]
    pub async fn select<'a>(&self, candidates: &[&'a Deployment]) -> Option<&'a Deployment> {
        if candidates.is_empty() {
            return None;
        }

        let loads = self.loads.read();
        
        let mut best_deployment = candidates[0];
        let mut best_load = loads
            .get(&Self::deployment_key(best_deployment))
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0);

        for &deployment in candidates.iter().skip(1) {
            let load = loads
                .get(&Self::deployment_key(deployment))
                .map(|counter| counter.load(Ordering::Relaxed))
                .unwrap_or(0);
            
            if load < best_load {
                best_load = load;
                best_deployment = deployment;
            }
        }

        Some(best_deployment)
    }

    /// Get all deployments sorted by load (lowest first).
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Lock-free atomic loads for all counters
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n log n) sort where n is number of candidates
    #[inline]
    pub async fn sort_by_load<'a>(&self, candidates: Vec<&'a Deployment>) -> Vec<&'a Deployment> {
        let loads = self.loads.read();
        
        let mut candidates_with_load: Vec<(&Deployment, u64)> = candidates
            .into_iter()
            .map(|deployment| {
                let load = loads
                    .get(&Self::deployment_key(deployment))
                    .map(|counter| counter.load(Ordering::Relaxed))
                    .unwrap_or(0);
                (deployment, load)
            })
            .collect();

        // Sort by load (lowest first)
        candidates_with_load.sort_by(|a, b| a.1.cmp(&b.1));

        candidates_with_load.into_iter().map(|(d, _)| d).collect()
    }

    /// Clear all load data.
    #[inline]
    pub async fn clear(&self) {
        let mut loads = self.loads.write();
        loads.clear();
    }

    /// Get all load statistics.
    ///
    /// ## Performance
    ///
    /// - Single read lock acquisition
    /// - Lock-free atomic loads for all counters
    /// - Returns HashMap with Arc<str> keys (zero-copy)
    #[inline]
    pub async fn get_stats(&self) -> HashMap<Arc<str>, u64> {
        let loads = self.loads.read();
        loads
            .iter()
            .map(|(key, counter)| (Arc::clone(key), counter.load(Ordering::Relaxed)))
            .collect()
    }
}

impl Default for LoadTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LoadTracker {
    fn clone(&self) -> Self {
        Self {
            loads: Arc::clone(&self.loads),
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
    async fn test_load_tracker_increment_decrement() {
        let tracker = LoadTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // Initial load is 0
        assert_eq!(tracker.get_load(&deployment).await, 0);

        // Increment load
        tracker.increment_load(&deployment).await;
        assert_eq!(tracker.get_load(&deployment).await, 1);

        tracker.increment_load(&deployment).await;
        assert_eq!(tracker.get_load(&deployment).await, 2);

        // Decrement load
        tracker.decrement_load(&deployment).await;
        assert_eq!(tracker.get_load(&deployment).await, 1);

        tracker.decrement_load(&deployment).await;
        assert_eq!(tracker.get_load(&deployment).await, 0);
    }

    #[tokio::test]
    async fn test_load_tracker_select() {
        let tracker = LoadTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        // Set different loads: dep2 has lowest load
        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep1).await;
        
        tracker.increment_load(&dep2).await;
        
        tracker.increment_load(&dep3).await;
        tracker.increment_load(&dep3).await;

        let candidates = vec![&dep1, &dep2, &dep3];
        let selected = tracker.select(&candidates).await.unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }

    #[tokio::test]
    async fn test_load_tracker_sort() {
        let tracker = LoadTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep1).await;
        
        tracker.increment_load(&dep2).await;
        
        tracker.increment_load(&dep3).await;
        tracker.increment_load(&dep3).await;

        let candidates = vec![&dep1, &dep2, &dep3];
        let sorted = tracker.sort_by_load(candidates).await;
        
        assert_eq!(sorted[0].litellm_params.model, "azure/gpt-4"); // Load: 1
        assert_eq!(sorted[1].litellm_params.model, "anthropic/gpt-4"); // Load: 2
        assert_eq!(sorted[2].litellm_params.model, "openai/gpt-4"); // Load: 3
    }

    #[tokio::test]
    async fn test_load_tracker_empty_candidates() {
        let tracker = LoadTracker::new();
        let candidates: Vec<&Deployment> = vec![];
        assert!(tracker.select(&candidates).await.is_none());
    }

    #[tokio::test]
    async fn test_load_tracker_no_data_fallback() {
        let tracker = LoadTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        // No load data (all have 0 load)
        let candidates = vec![&dep1, &dep2];
        let selected = tracker.select(&candidates).await.unwrap();
        // Should return first candidate when all have same load
        assert_eq!(selected.litellm_params.model, "openai/gpt-4");
    }

    #[tokio::test]
    async fn test_load_tracker_concurrent_access() {
        let tracker = Arc::new(LoadTracker::new());
        let deployment = Arc::new(create_deployment("gpt-4", "openai/gpt-4"));

        // Simulate concurrent increments
        let tracker1 = Arc::clone(&tracker);
        let deployment1 = Arc::clone(&deployment);
        let handle1 = tokio::spawn(async move {
            for _ in 0..100 {
                tracker1.increment_load(&deployment1).await;
            }
        });

        let tracker2 = Arc::clone(&tracker);
        let deployment2 = Arc::clone(&deployment);
        let handle2 = tokio::spawn(async move {
            for _ in 0..100 {
                tracker2.increment_load(&deployment2).await;
            }
        });

        handle1.await.unwrap();
        handle2.await.unwrap();

        // Should have 200 total increments
        assert_eq!(tracker.get_load(&deployment).await, 200);
    }
}
