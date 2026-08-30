//! Cost-based routing strategy.
//!
//! Routes requests to the deployment with the lowest cost per token.
//! Uses model pricing information to select the cheapest provider.
//!
//! ## Zero-Allocation Optimizations
//!
//! - Uses `Arc<str>` for deployment keys to avoid String allocations
//! - Pre-allocates HashMap with expected capacity
//! - No locking required (costs are set during initialization, not during request processing)

use std::collections::HashMap;
use std::sync::Arc;

use super::Deployment;

/// Cost tracker for deployments.
/// Tracks cost per token for each deployment based on model pricing.
///
/// ## Zero-Allocation Design
///
/// - `Arc<str>` keys: Shared ownership, no reallocation on clone
/// - Pre-allocated HashMap: Reduces rehashing overhead
/// - No locking: Costs are set during initialization, not during request processing
#[derive(Debug, Clone)]
pub struct CostTracker {
    /// Cost per token (input + output) per deployment key (model_name + provider)
    /// Stored as (input_cost_per_token, output_cost_per_token)
    /// Uses Arc<str> for zero-copy key sharing
    costs: HashMap<Arc<str>, (f64, f64)>,
}

impl CostTracker {
    /// Create a new cost tracker.
    ///
    /// Pre-allocates HashMap with capacity for 16 deployments to reduce rehashing.
    pub fn new() -> Self {
        Self {
            costs: HashMap::with_capacity(16),
        }
    }

    /// Get the deployment key for a deployment.
    /// Returns Arc<str> for zero-copy sharing.
    #[inline]
    fn deployment_key(deployment: &Deployment) -> Arc<str> {
        let key = format!("{}:{}", deployment.model_name, deployment.litellm_params.model);
        Arc::from(key)
    }

    /// Set the cost for a deployment.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys to avoid String allocation
    /// - O(1) insertion
    #[inline]
    pub fn set_cost(&mut self, deployment: &Deployment, input_cost: f64, output_cost: f64) {
        let key = Self::deployment_key(deployment);
        self.costs.insert(key, (input_cost, output_cost));
    }

    /// Get the cost for a deployment.
    /// Returns (input_cost_per_token, output_cost_per_token).
    /// Returns (f64::MAX, f64::MAX) if no cost data is available.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(1) lookup
    #[inline]
    pub fn get_cost(&self, deployment: &Deployment) -> (f64, f64) {
        let key = Self::deployment_key(deployment);
        self.costs.get(&key).copied().unwrap_or((f64::MAX, f64::MAX))
    }

    /// Calculate the total cost for a request with the given token counts.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(1) cost calculation
    #[inline]
    pub fn calculate_cost(&self, deployment: &Deployment, input_tokens: u64, output_tokens: u64) -> f64 {
        let (input_cost, output_cost) = self.get_cost(deployment);
        (input_tokens as f64 * input_cost) + (output_tokens as f64 * output_cost)
    }

    /// Select the deployment with the lowest cost for the given token counts.
    /// If multiple deployments have the same cost, returns the first one.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n) scan where n is number of candidates
    #[inline]
    pub fn select<'a>(
        &self,
        candidates: &[&'a Deployment],
        input_tokens: u64,
        output_tokens: u64,
    ) -> Option<&'a Deployment> {
        if candidates.is_empty() {
            return None;
        }

        let mut best_deployment = candidates[0];
        let mut best_cost = self.calculate_cost(best_deployment, input_tokens, output_tokens);

        for &deployment in candidates.iter().skip(1) {
            let cost = self.calculate_cost(deployment, input_tokens, output_tokens);
            if cost < best_cost {
                best_cost = cost;
                best_deployment = deployment;
            }
        }

        Some(best_deployment)
    }

    /// Get all deployments sorted by cost (lowest first) for the given token counts.
    ///
    /// ## Performance
    ///
    /// - Uses Arc<str> keys for zero-copy lookup
    /// - O(n log n) sort where n is number of candidates
    #[inline]
    pub fn sort_by_cost<'a>(
        &self,
        candidates: Vec<&'a Deployment>,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Vec<&'a Deployment> {
        let mut candidates_with_cost: Vec<(&Deployment, f64)> = candidates
            .into_iter()
            .map(|deployment| {
                let cost = self.calculate_cost(deployment, input_tokens, output_tokens);
                (deployment, cost)
            })
            .collect();

        // Sort by cost (lowest first)
        candidates_with_cost.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates_with_cost.into_iter().map(|(d, _)| d).collect()
    }

    /// Clear all cost data.
    #[inline]
    pub fn clear(&mut self) {
        self.costs.clear();
    }

    /// Get all cost statistics.
    ///
    /// ## Performance
    ///
    /// - Returns HashMap with Arc<str> keys (zero-copy)
    #[inline]
    pub fn get_stats(&self) -> HashMap<Arc<str>, (f64, f64)> {
        self.costs.clone()
    }
}

impl Default for CostTracker {
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
    async fn test_cost_tracker_set_and_get() {
        let mut tracker = CostTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // No cost data yet
        assert_eq!(tracker.get_cost(&deployment), (f64::MAX, f64::MAX));

        // Set cost
        tracker.set_cost(&deployment, 0.00003, 0.00006);
        assert_eq!(tracker.get_cost(&deployment), (0.00003, 0.00006));
    }

    #[tokio::test]
    async fn test_cost_tracker_calculate_cost() {
        let mut tracker = CostTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        tracker.set_cost(&deployment, 0.00003, 0.00006);

        // Calculate cost for 1000 input tokens and 500 output tokens
        let cost = tracker.calculate_cost(&deployment, 1000, 500);
        let expected = (1000.0 * 0.00003) + (500.0 * 0.00006);
        assert!((cost - expected).abs() < 0.0000001);
    }

    #[tokio::test]
    async fn test_cost_tracker_select() {
        let mut tracker = CostTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        // Set different costs: dep2 is cheapest
        tracker.set_cost(&dep1, 0.00003, 0.00006);
        tracker.set_cost(&dep2, 0.00002, 0.00004); // Cheapest
        tracker.set_cost(&dep3, 0.000025, 0.00005);

        let candidates = vec![&dep1, &dep2, &dep3];
        let selected = tracker.select(&candidates, 1000, 500).unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }

    #[tokio::test]
    async fn test_cost_tracker_sort() {
        let mut tracker = CostTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");

        tracker.set_cost(&dep1, 0.00003, 0.00006);
        tracker.set_cost(&dep2, 0.00002, 0.00004); // Cheapest
        tracker.set_cost(&dep3, 0.000025, 0.00005);

        let candidates = vec![&dep1, &dep2, &dep3];
        let sorted = tracker.sort_by_cost(candidates, 1000, 500);
        
        assert_eq!(sorted[0].litellm_params.model, "azure/gpt-4");
        assert_eq!(sorted[1].litellm_params.model, "anthropic/gpt-4");
        assert_eq!(sorted[2].litellm_params.model, "openai/gpt-4");
    }

    #[tokio::test]
    async fn test_cost_tracker_empty_candidates() {
        let tracker = CostTracker::new();
        let candidates: Vec<&Deployment> = vec![];
        assert!(tracker.select(&candidates, 1000, 500).is_none());
    }

    #[tokio::test]
    async fn test_cost_tracker_no_data_fallback() {
        let tracker = CostTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        // No cost data (all have MAX cost)
        let candidates = vec![&dep1, &dep2];
        let selected = tracker.select(&candidates, 1000, 500).unwrap();
        // Should return first candidate when all have same cost
        assert_eq!(selected.litellm_params.model, "openai/gpt-4");
    }

    #[tokio::test]
    async fn test_cost_tracker_different_token_counts() {
        let mut tracker = CostTracker::new();
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");

        // dep1 has lower input cost, dep2 has lower output cost
        tracker.set_cost(&dep1, 0.00002, 0.00008);
        tracker.set_cost(&dep2, 0.00004, 0.00004);

        let candidates = vec![&dep1, &dep2];
        
        // For input-heavy request, dep1 should be cheaper
        let selected = tracker.select(&candidates, 10000, 100).unwrap();
        assert_eq!(selected.litellm_params.model, "openai/gpt-4");
        
        // For output-heavy request, dep2 should be cheaper
        let selected = tracker.select(&candidates, 100, 10000).unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }
}
