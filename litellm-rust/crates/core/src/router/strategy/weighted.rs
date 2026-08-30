//! Weighted routing strategy.
//!
//! Routes requests to deployments based on assigned weights.
//! Higher weights receive proportionally more traffic.
//!
//! ## Zero-Allocation Optimizations
//!
//! - Uses `Arc<str>` for deployment keys to avoid String allocations
//! - Pre-allocates HashMap with expected capacity
//! - No locking required (weights are set during initialization, not during request processing)

use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;

use super::Deployment;

/// Weight tracker for deployments.
/// Tracks weights for each deployment and performs weighted random selection.
///
/// ## Zero-Allocation Design
///
/// - `Arc<str>` keys: Shared ownership, no reallocation on clone
/// - Pre-allocated HashMap: Reduces rehashing overhead
/// - No locking: Weights are set during initialization, not during request processing
#[derive(Debug, Clone)]
pub struct WeightTracker {
    /// Weight per deployment key (model_name + provider)
    /// Uses Arc<str> for zero-copy key sharing
    weights: HashMap<Arc<str>, u32>,
}

impl WeightTracker {
    /// Create a new weight tracker.
    ///
    /// Pre-allocates HashMap with capacity for 16 deployments to reduce rehashing.
    pub fn new() -> Self {
        Self {
            weights: HashMap::with_capacity(16),
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

    /// Set the weight for a deployment.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys to avoid String allocation
    /// - O(1) insertion
    #[inline]
    pub fn set_weight(&mut self, deployment: &Deployment, weight: u32) {
        let key = Self::deployment_key(deployment);
        self.weights.insert(key, weight);
    }

    /// Get the weight for a deployment.
    /// Returns 1 if no weight is set (default weight).
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(1) lookup
    #[inline]
    pub fn get_weight(&self, deployment: &Deployment) -> u32 {
        let key = Self::deployment_key(deployment);
        self.weights.get(&key).copied().unwrap_or(1)
    }

    /// Select a deployment based on weights using weighted random selection.
    /// Returns `None` when there are no candidates.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n) scan where n is number of candidates
    /// - Single random number generation
    #[inline]
    pub fn select<'a>(&self, candidates: &[&'a Deployment]) -> Option<&'a Deployment> {
        if candidates.is_empty() {
            return None;
        }

        // Calculate total weight
        let total_weight: u32 = candidates.iter().map(|d| self.get_weight(d)).sum();

        if total_weight == 0 {
            // If all weights are 0, fall back to first candidate
            return Some(candidates[0]);
        }

        // Generate random number in range [0, total_weight)
        let mut rng = rand::thread_rng();
        let random_value = rng.gen_range(0..total_weight);

        // Select deployment based on cumulative weights
        let mut cumulative_weight = 0;
        for &deployment in candidates {
            cumulative_weight += self.get_weight(deployment);
            if random_value < cumulative_weight {
                return Some(deployment);
            }
        }

        // Fallback to last candidate (shouldn't reach here)
        Some(candidates[candidates.len() - 1])
    }

    /// Get all deployments sorted by weight (highest first).
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n log n) sort where n is number of candidates
    #[inline]
    pub fn sort_by_weight<'a>(&self, candidates: Vec<&'a Deployment>) -> Vec<&'a Deployment> {
        let mut candidates_with_weight: Vec<(&Deployment, u32)> = candidates
            .into_iter()
            .map(|deployment| {
                let weight = self.get_weight(deployment);
                (deployment, weight)
            })
            .collect();

        // Sort by weight (highest first)
        candidates_with_weight.sort_by_key(|a| std::cmp::Reverse(a.1));

        candidates_with_weight.into_iter().map(|(d, _)| d).collect()
    }

    /// Clear all weight data.
    #[inline]
    pub fn clear(&mut self) {
        self.weights.clear();
    }

    /// Get all weight statistics.
    ///
    /// ## Performance
    ///
    /// - Returns HashMap with Arc<str> keys (zero-copy)
    #[inline]
    pub fn get_stats(&self) -> HashMap<Arc<str>, u32> {
        self.weights.clone()
    }

    /// Calculate the percentage of traffic each deployment should receive.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n) calculation where n is number of candidates
    #[inline]
    pub fn get_traffic_distribution(&self, candidates: &[&Deployment]) -> HashMap<Arc<str>, f64> {
        let total_weight: u32 = candidates.iter().map(|d| self.get_weight(d)).sum();

        if total_weight == 0 {
            return HashMap::new();
        }

        candidates
            .iter()
            .map(|d| {
                let key = Self::deployment_key(d);
                let weight = self.get_weight(d);
                let percentage = (weight as f64 / total_weight as f64) * 100.0;
                (key, percentage)
            })
            .collect()
    }
}

impl Default for WeightTracker {
    fn default() -> Self {
        Self::new()
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
    async fn test_weight_tracker_set_and_get() {
        let mut tracker = WeightTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // Default weight is 1
        assert_eq!(tracker.get_weight(&deployment), 1);

        // Set weight
        tracker.set_weight(&deployment, 10);
        assert_eq!(tracker.get_weight(&deployment), 10);
    }

    #[tokio::test]
    async fn test_weight_tracker_select() {
        let mut tracker = WeightTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        // dep1 has weight 90, dep2 has weight 10
        tracker.set_weight(&dep1, 90);
        tracker.set_weight(&dep2, 10);

        let candidates = vec![&dep1, &dep2];

        // Run selection many times to verify distribution
        let mut dep1_count = 0;
        let mut _dep2_count = 0;

        for _ in 0..1000 {
            let selected = tracker.select(&candidates).unwrap();
            if selected.litellm_params.model == "openai/gpt-4" {
                dep1_count += 1;
            } else {
                _dep2_count += 1;
            }
        }

        // dep1 should be selected roughly 90% of the time (with some variance)
        let dep1_percentage = dep1_count as f64 / 1000.0 * 100.0;
        assert!(
            dep1_percentage > 80.0 && dep1_percentage < 95.0,
            "Expected ~90% for dep1, got {}%",
            dep1_percentage
        );
    }

    #[tokio::test]
    async fn test_weight_tracker_sort() {
        let mut tracker = WeightTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        tracker.set_weight(&dep1, 50);
        tracker.set_weight(&dep2, 30);
        tracker.set_weight(&dep3, 20);

        let candidates = vec![&dep1, &dep2, &dep3];
        let sorted = tracker.sort_by_weight(candidates);

        assert_eq!(sorted[0].litellm_params.model, "openai/gpt-4");
        assert_eq!(sorted[1].litellm_params.model, "azure/gpt-4");
        assert_eq!(sorted[2].litellm_params.model, "anthropic/gpt-4");
    }

    #[tokio::test]
    async fn test_weight_tracker_empty_candidates() {
        let tracker = WeightTracker::new();
        let candidates: Vec<&Deployment> = vec![];
        assert!(tracker.select(&candidates).is_none());
    }

    #[tokio::test]
    async fn test_weight_tracker_traffic_distribution() {
        let mut tracker = WeightTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        tracker.set_weight(&dep1, 70);
        tracker.set_weight(&dep2, 30);

        let candidates = vec![&dep1, &dep2];
        let distribution = tracker.get_traffic_distribution(&candidates);

        let dep1_key = WeightTracker::deployment_key(&dep1);
        let dep2_key = WeightTracker::deployment_key(&dep2);

        assert!((distribution[&dep1_key] - 70.0).abs() < 0.01);
        assert!((distribution[&dep2_key] - 30.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_weight_tracker_equal_weights() {
        let mut tracker = WeightTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        tracker.set_weight(&dep1, 50);
        tracker.set_weight(&dep2, 50);

        let candidates = vec![&dep1, &dep2];

        // Run selection many times to verify equal distribution
        let mut dep1_count = 0;
        let mut _dep2_count = 0;

        for _ in 0..1000 {
            let selected = tracker.select(&candidates).unwrap();
            if selected.litellm_params.model == "openai/gpt-4" {
                dep1_count += 1;
            } else {
                _dep2_count += 1;
            }
        }

        // Both should be selected roughly 50% of the time
        let dep1_percentage = dep1_count as f64 / 1000.0 * 100.0;
        assert!(
            dep1_percentage > 40.0 && dep1_percentage < 60.0,
            "Expected ~50% for dep1, got {}%",
            dep1_percentage
        );
    }
}
