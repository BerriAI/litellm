//! Minimal Rust port of LiteLLM's `router.py` deployment selection.
//!
//! A [`Router`] is built from a `model_list` of [`Deployment`]s
//! (`{ model_name, litellm_params: { model, api_key, api_base } }`) and selects
//! one per request via a [`RoutingStrategy`]. For now the only strategy is
//! `simple-shuffle` — a uniform random pick within a `model_name` group.
//!
//! This stays pure (no I/O): it only *chooses* a deployment. The host (the
//! gateway) takes the chosen deployment and performs the actual provider call.
//!
//! - [`deployment`] — the `model_list` data types.
//! - [`strategy`] — how a deployment is chosen.
//! - [`health_monitor`] — tracks deployment health metrics.

mod deployment;
mod health_monitor;
mod strategy;

pub use deployment::{Deployment, LiteLLMParams};
pub use health_monitor::{HealthConfig, HealthMetrics, HealthMonitor, HealthStatus};
pub use strategy::{CostTracker, LatencyTracker, LoadTracker, RoutingStrategy, WeightTracker};

use std::sync::Arc;

/// Load-balancing router over a `model_list`.
#[derive(Clone, Debug)]
pub struct Router {
    model_list: Vec<Deployment>,
    routing_strategy: RoutingStrategy,
    latency_tracker: Option<Arc<LatencyTracker>>,
    load_tracker: Option<Arc<LoadTracker>>,
    cost_tracker: Option<Arc<CostTracker>>,
    weight_tracker: Option<Arc<WeightTracker>>,
    health_monitor: Arc<HealthMonitor>,
}

impl Default for Router {
    fn default() -> Self {
        Self {
            model_list: Vec::new(),
            routing_strategy: RoutingStrategy::default(),
            latency_tracker: None,
            load_tracker: None,
            cost_tracker: None,
            weight_tracker: None,
            health_monitor: Arc::new(HealthMonitor::default()),
        }
    }
}

impl Router {
    /// Build a router from a `model_list` using the default `simple-shuffle` strategy.
    pub fn new(model_list: Vec<Deployment>) -> Self {
        Self {
            model_list,
            routing_strategy: RoutingStrategy::SimpleShuffle,
            latency_tracker: None,
            load_tracker: None,
            cost_tracker: None,
            weight_tracker: None,
            health_monitor: Arc::new(HealthMonitor::default()),
        }
    }

    /// Build a router with a specific routing strategy.
    pub fn with_strategy(model_list: Vec<Deployment>, strategy: RoutingStrategy) -> Self {
        let latency_tracker = if strategy == RoutingStrategy::LatencyBased {
            Some(Arc::new(LatencyTracker::new(100))) // Window size of 100 samples
        } else {
            None
        };

        let load_tracker = if strategy == RoutingStrategy::LoadBased {
            Some(Arc::new(LoadTracker::new()))
        } else {
            None
        };

        let cost_tracker = if strategy == RoutingStrategy::CostBased {
            Some(Arc::new(CostTracker::new()))
        } else {
            None
        };

        let weight_tracker = if strategy == RoutingStrategy::Weighted {
            Some(Arc::new(WeightTracker::new()))
        } else {
            None
        };

        Self {
            model_list,
            routing_strategy: strategy,
            latency_tracker,
            load_tracker,
            cost_tracker,
            weight_tracker,
            health_monitor: Arc::new(HealthMonitor::default()),
        }
    }

    /// Build a router with a specific routing strategy and health config.
    pub fn with_strategy_and_health(
        model_list: Vec<Deployment>,
        strategy: RoutingStrategy,
        health_config: HealthConfig,
    ) -> Self {
        let mut router = Self::with_strategy(model_list, strategy);
        router.health_monitor = Arc::new(HealthMonitor::new(health_config));
        router
    }

    /// Get the routing strategy.
    pub fn strategy(&self) -> RoutingStrategy {
        self.routing_strategy
    }

    /// Get the latency tracker if available.
    pub fn latency_tracker(&self) -> Option<&Arc<LatencyTracker>> {
        self.latency_tracker.as_ref()
    }

    /// Get the load tracker if available.
    pub fn load_tracker(&self) -> Option<&Arc<LoadTracker>> {
        self.load_tracker.as_ref()
    }

    /// Get the cost tracker if available.
    pub fn cost_tracker(&self) -> Option<&Arc<CostTracker>> {
        self.cost_tracker.as_ref()
    }

    /// Get the weight tracker if available.
    pub fn weight_tracker(&self) -> Option<&Arc<WeightTracker>> {
        self.weight_tracker.as_ref()
    }

    /// Get the health monitor.
    pub fn health_monitor(&self) -> &Arc<HealthMonitor> {
        &self.health_monitor
    }

    /// All deployments in the `model_list`. Read-only; used by the host to
    /// enumerate upstreams (e.g. to pre-warm a connection pool per deployment).
    pub fn deployments(&self) -> &[Deployment] {
        &self.model_list
    }

    /// Whether any deployment is registered under `model`.
    pub fn has_deployment(&self, model: &str) -> bool {
        self.model_list
            .iter()
            .any(|deployment| deployment.model_name == model)
    }

    /// Pick a deployment for `model` per the routing strategy. Returns `None`
    /// when no deployment is registered under that `model_name`.
    ///
    /// Note: For latency-based routing, use `get_available_deployment_with_latency` instead.
    pub fn get_available_deployment(&self, model: &str) -> Option<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();
        self.routing_strategy.select(&candidates)
    }

    /// Pick a deployment for `model` using latency-based routing (async).
    /// Returns `None` when no deployment is registered or no latency tracker is available.
    pub async fn get_available_deployment_with_latency(&self, model: &str) -> Option<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        match self.routing_strategy {
            RoutingStrategy::LatencyBased => {
                if let Some(tracker) = &self.latency_tracker {
                    tracker.select(&candidates).await
                } else {
                    // Fallback to simple shuffle if no tracker
                    self.routing_strategy.select(&candidates)
                }
            }
            _ => self.routing_strategy.select(&candidates),
        }
    }

    /// Pick a deployment for `model` using load-based routing (async).
    /// Returns `None` when no deployment is registered or no load tracker is available.
    pub async fn get_available_deployment_with_load(&self, model: &str) -> Option<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        match self.routing_strategy {
            RoutingStrategy::LoadBased => {
                if let Some(tracker) = &self.load_tracker {
                    tracker.select(&candidates).await
                } else {
                    // Fallback to simple shuffle if no tracker
                    self.routing_strategy.select(&candidates)
                }
            }
            _ => self.routing_strategy.select(&candidates),
        }
    }

    /// Get all deployments for `model`, ordered by routing strategy.
    /// Used for fallback routing where we try deployments in order.
    pub fn get_all_deployments(&self, model: &str) -> Vec<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        // For now, just return in the order they appear in model_list
        // Future: apply routing strategy ordering (latency-based, etc.)
        candidates
    }

    /// Get all deployments for `model`, ordered by latency (async).
    /// Returns deployments sorted by average latency (lowest first).
    pub async fn get_all_deployments_by_latency(&self, model: &str) -> Vec<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if let Some(tracker) = &self.latency_tracker {
            tracker.sort_by_latency(candidates).await
        } else {
            candidates
        }
    }

    /// Get all deployments for `model`, ordered by load (async).
    /// Returns deployments sorted by current load (lowest first).
    pub async fn get_all_deployments_by_load(&self, model: &str) -> Vec<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if let Some(tracker) = &self.load_tracker {
            tracker.sort_by_load(candidates).await
        } else {
            candidates
        }
    }

    /// Pick a deployment for `model` using cost-based routing.
    /// Returns `None` when no deployment is registered or no cost tracker is available.
    /// Requires estimated token counts to calculate costs.
    pub fn get_available_deployment_with_cost(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Option<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        match self.routing_strategy {
            RoutingStrategy::CostBased => {
                if let Some(tracker) = &self.cost_tracker {
                    tracker.select(&candidates, input_tokens, output_tokens)
                } else {
                    // Fallback to simple shuffle if no tracker
                    self.routing_strategy.select(&candidates)
                }
            }
            _ => self.routing_strategy.select(&candidates),
        }
    }

    /// Get all deployments for `model`, ordered by cost.
    /// Returns deployments sorted by cost for the given token counts (lowest first).
    pub fn get_all_deployments_by_cost(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Vec<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if let Some(tracker) = &self.cost_tracker {
            tracker.sort_by_cost(candidates, input_tokens, output_tokens)
        } else {
            candidates
        }
    }

    /// Pick a deployment for `model` using weighted routing.
    /// Returns `None` when no deployment is registered or no weight tracker is available.
    pub fn get_available_deployment_with_weight(&self, model: &str) -> Option<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        match self.routing_strategy {
            RoutingStrategy::Weighted => {
                if let Some(tracker) = &self.weight_tracker {
                    tracker.select(&candidates)
                } else {
                    // Fallback to simple shuffle if no tracker
                    self.routing_strategy.select(&candidates)
                }
            }
            _ => self.routing_strategy.select(&candidates),
        }
    }

    /// Get all deployments for `model`, ordered by weight (highest first).
    pub fn get_all_deployments_by_weight(&self, model: &str) -> Vec<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();

        if let Some(tracker) = &self.weight_tracker {
            tracker.sort_by_weight(candidates)
        } else {
            candidates
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deployment(name: &str, model: &str) -> Deployment {
        Deployment {
            model_name: name.to_string(),
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

    #[test]
    fn selects_a_matching_deployment() {
        let router = Router::new(vec![
            deployment("gpt-realtime", "gpt-realtime"),
            deployment("other", "other-model"),
        ]);
        let chosen = router
            .get_available_deployment("gpt-realtime")
            .expect("a deployment should match");
        assert_eq!(chosen.model_name, "gpt-realtime");
    }

    #[test]
    fn unknown_model_returns_none() {
        let router = Router::new(vec![deployment("gpt-realtime", "gpt-realtime")]);
        assert!(router.get_available_deployment("missing").is_none());
    }

    #[test]
    fn get_all_deployments_returns_all_matching() {
        let router = Router::new(vec![
            deployment("gpt-4", "openai/gpt-4-turbo"),
            deployment("gpt-4", "openai/gpt-4"),
            deployment("gpt-3.5", "openai/gpt-3.5-turbo"),
        ]);
        let all_gpt4 = router.get_all_deployments("gpt-4");
        assert_eq!(all_gpt4.len(), 2);
        assert_eq!(all_gpt4[0].litellm_params.model, "openai/gpt-4-turbo");
        assert_eq!(all_gpt4[1].litellm_params.model, "openai/gpt-4");
    }

    #[test]
    fn get_all_deployments_empty_for_unknown_model() {
        let router = Router::new(vec![deployment("gpt-4", "openai/gpt-4")]);
        let all = router.get_all_deployments("missing");
        assert!(all.is_empty());
    }
}
