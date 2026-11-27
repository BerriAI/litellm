//! Which routing strategy picks for a request, and whose state it reads: a port of
//! `Router._get_routing_context`. A per-request `routing_strategy` override wins, then the
//! routing group the resolved model belongs to (called as the group, or as a member of a
//! non-priority group), then the router's own strategy.

use litellm_router_types::RoutingStrategy;

use crate::registry::{Registry, ResolveKind};

/// Whose selector state a strategy pick reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrategyScope {
    /// The router's own strategy (the implicit `"default"` group).
    Default,
    /// A routing group's own selector.
    Group(String),
    /// A per-request override: the default selector when it names the router's strategy, else
    /// one selector per strategy kept for the router's life.
    Override,
}

impl StrategyScope {
    /// The scope as the host step spells it.
    pub fn wire(&self) -> (&'static str, Option<&str>) {
        match self {
            Self::Default => ("default", None),
            Self::Group(name) => ("group", Some(name)),
            Self::Override => ("override", None),
        }
    }
}

/// Whether router-core picks by `strategy` itself. Every other strategy is a
/// [`crate::RouterHost::select`] step until router-core implements it.
pub fn implemented(strategy: &RoutingStrategy) -> bool {
    matches!(
        strategy,
        RoutingStrategy::SimpleShuffle | RoutingStrategy::Unset
    )
}

/// A request's `routing_strategy` override, when it is one `_OVERRIDABLE_ROUTING_STRATEGIES`
/// honours: simple-shuffle and every strategy with a default selector.
pub fn request_override(raw: Option<&str>) -> Option<RoutingStrategy> {
    match raw?.parse().ok()? {
        strategy @ (RoutingStrategy::SimpleShuffle
        | RoutingStrategy::LeastBusy
        | RoutingStrategy::UsageBased
        | RoutingStrategy::UsageBasedV2
        | RoutingStrategy::LatencyBased
        | RoutingStrategy::CostBased) => Some(strategy),
        RoutingStrategy::ProviderBudget
        | RoutingStrategy::Lar1
        | RoutingStrategy::Priority
        | RoutingStrategy::Unset
        | RoutingStrategy::Other(_) => None,
    }
}

/// The strategy and selector scope for a pick of `model_group` (the resolved group, after
/// aliases), reached through a resolution of `kind`.
pub fn routing_context(
    registry: &Registry,
    default: &RoutingStrategy,
    override_strategy: Option<&str>,
    kind: &ResolveKind,
    model_group: &str,
) -> (RoutingStrategy, StrategyScope) {
    if let Some(strategy) = request_override(override_strategy) {
        return (strategy, StrategyScope::Override);
    }
    let group = match kind {
        ResolveKind::RoutingGroup => registry
            .routing_groups()
            .iter()
            .find(|group| group.group_name == model_group),
        ResolveKind::Alias | ResolveKind::Group => registry.member_routing_group(model_group),
        ResolveKind::SpecificDeployment
        | ResolveKind::Id
        | ResolveKind::Default
        | ResolveKind::Wildcard => None,
    };
    match group {
        None => (default.clone(), StrategyScope::Default),
        Some(group) if crate::routing_groups::is_priority(group) => {
            (RoutingStrategy::SimpleShuffle, StrategyScope::Default)
        }
        Some(group) => (
            group.routing_strategy.clone(),
            StrategyScope::Group(group.group_name.clone()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::honoured(Some("least-busy"), Some(RoutingStrategy::LeastBusy))]
    #[case::shuffle(Some("simple-shuffle"), Some(RoutingStrategy::SimpleShuffle))]
    #[case::usage_based(Some("usage-based-routing"), Some(RoutingStrategy::UsageBased))]
    #[case::usage_based_v2(Some("usage-based-routing-v2"), Some(RoutingStrategy::UsageBasedV2))]
    #[case::latency_based(Some("latency-based-routing"), Some(RoutingStrategy::LatencyBased))]
    #[case::cost_based(Some("cost-based-routing"), Some(RoutingStrategy::CostBased))]
    #[case::lar1_ignored(Some("lar1"), None)]
    #[case::priority_ignored(Some("priority"), None)]
    #[case::provider_budget_ignored(Some("provider-budget-routing"), None)]
    #[case::none_ignored(Some("None"), None)]
    #[case::unknown_ignored(Some("bogus"), None)]
    #[case::unset(None, None)]
    fn only_overridable_strategies_override(
        #[case] raw: Option<&str>,
        #[case] expected: Option<RoutingStrategy>,
    ) {
        assert_eq!(request_override(raw), expected);
    }
}
