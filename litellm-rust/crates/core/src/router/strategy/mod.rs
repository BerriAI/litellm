//! Routing policy: how the router picks one deployment from a model group.
//!
//! One module per strategy; [`RoutingStrategy::select`] dispatches to it. New
//! strategies (least-busy, latency-based, …) get their own file here.

mod cost_based;
mod latency_based;
mod load_based;
mod simple_shuffle;
mod weighted;

use super::Deployment;

pub use cost_based::CostTracker;
pub use latency_based::LatencyTracker;
pub use load_based::LoadTracker;
pub use weighted::WeightTracker;

/// How the router chooses among the deployments sharing a `model_name`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Uniform random pick among the matching deployments.
    #[default]
    SimpleShuffle,
    /// Route to the deployment with the lowest average latency.
    LatencyBased,
    /// Route to the deployment with the lowest current load.
    LoadBased,
    /// Route to the deployment with the lowest cost.
    CostBased,
    /// Route to deployments based on assigned weights.
    Weighted,
}

impl RoutingStrategy {
    /// Choose one deployment from `candidates` (all sharing the requested
    /// `model_name`). Returns `None` when there are no candidates.
    pub fn select<'a>(&self, candidates: &[&'a Deployment]) -> Option<&'a Deployment> {
        match self {
            RoutingStrategy::SimpleShuffle => simple_shuffle::select(candidates),
            RoutingStrategy::LatencyBased => {
                // Latency-based routing requires async context and a LatencyTracker.
                // This synchronous select falls back to simple shuffle.
                // Use Router::get_available_deployment_with_latency for async selection.
                simple_shuffle::select(candidates)
            }
            RoutingStrategy::LoadBased => {
                // Load-based routing requires async context and a LoadTracker.
                // This synchronous select falls back to simple shuffle.
                // Use Router::get_available_deployment_with_load for async selection.
                simple_shuffle::select(candidates)
            }
            RoutingStrategy::CostBased => {
                // Cost-based routing requires a CostTracker and token counts.
                // This synchronous select falls back to simple shuffle.
                // Use Router::get_available_deployment_with_cost for async selection.
                simple_shuffle::select(candidates)
            }
            RoutingStrategy::Weighted => {
                // Weighted routing requires a WeightTracker.
                // This synchronous select falls back to simple shuffle.
                // Use Router::get_available_deployment_with_weight for selection.
                simple_shuffle::select(candidates)
            }
        }
    }
}
