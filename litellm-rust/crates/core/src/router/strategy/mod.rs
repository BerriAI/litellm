//! Routing policy: how the router picks one deployment from a model group.
//!
//! One module per strategy; [`RoutingStrategy::select`] dispatches to it. New
//! strategies (least-busy, latency-based, …) get their own file here.

mod simple_shuffle;

use super::Deployment;

/// How the router chooses among the deployments sharing a `model_name`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Uniform random pick among the matching deployments.
    #[default]
    SimpleShuffle,
}

impl RoutingStrategy {
    /// Choose one deployment from `candidates` (all sharing the requested
    /// `model_name`). Returns `None` when there are no candidates.
    pub fn select<'a>(&self, candidates: &[&'a Deployment]) -> Option<&'a Deployment> {
        match self {
            RoutingStrategy::SimpleShuffle => simple_shuffle::select(candidates),
        }
    }
}
