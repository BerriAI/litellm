//! Which routing strategy picks for a request, and whose state it reads: a port of
//! `Router._get_routing_context`. A per-request `routing_strategy` override wins, then the
//! routing group the resolved model belongs to (called as the group, or as a member of a
//! non-priority group), then the router's own strategy.

use crate::{
    config::RoutingStrategy,
    registry::{Registry, ResolveKind},
};

/// The per-request overrides `_OVERRIDABLE_ROUTING_STRATEGIES` honours: simple-shuffle and every
/// strategy with a default selector. Anything else is ignored.
pub const OVERRIDABLE: [&str; 6] = [
    "simple-shuffle",
    "least-busy",
    "usage-based-routing",
    "usage-based-routing-v2",
    "latency-based-routing",
    "cost-based-routing",
];

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

/// A request's `routing_strategy` override, when it is one Python honours.
pub fn request_override(raw: Option<&str>) -> Option<RoutingStrategy> {
    let raw = raw?;
    OVERRIDABLE
        .contains(&raw)
        .then(|| raw.parse().unwrap_or_default())
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
        // A priority group supplies order tiers, not a replacement selector.
        // Use the router's configured strategy to choose within the filtered tier.
        Some(group) if is_priority(&group.routing_strategy) => {
            (default.clone(), StrategyScope::Default)
        }
        Some(group) => (
            group.routing_strategy.clone(),
            StrategyScope::Group(group.group_name.clone()),
        ),
    }
}

fn is_priority(strategy: &RoutingStrategy) -> bool {
    matches!(strategy, RoutingStrategy::Other(name) if name == "priority")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::registry::{DeploymentSpec, RoutingGroupSpec};

    fn priority_registry() -> Registry {
        let deployments: Vec<DeploymentSpec> = ["m-a", "m-b"]
            .into_iter()
            .map(|model_name| {
                serde_json::from_value(json!({
                    "model_name": model_name,
                    "litellm_params": { "model": "openai/gpt-4o" },
                    "model_info": { "id": model_name },
                }))
                .unwrap()
            })
            .collect();

        let group = RoutingGroupSpec {
            group_name: "m-group".to_owned(),
            models: vec!["m-a".to_owned(), "m-b".to_owned()],
            routing_strategy: RoutingStrategy::Other("priority".to_owned()),
            model_priorities: Some(BTreeMap::from([
                ("m-a".to_owned(), 1),
                ("m-b".to_owned(), 1),
            ])),
        };

        Registry::build(deployments, BTreeMap::new(), vec![group]).unwrap()
    }

    #[rstest]
    #[case::usage_based_v2(RoutingStrategy::UsageBasedV2)]
    #[case::least_busy(RoutingStrategy::LeastBusy)]
    #[case::simple_shuffle(RoutingStrategy::SimpleShuffle)]
    fn priority_group_uses_router_default_strategy(#[case] default: RoutingStrategy) {
        let registry = priority_registry();

        let (strategy, scope) = routing_context(
            &registry,
            &default,
            None,
            &ResolveKind::RoutingGroup,
            "m-group",
        );

        assert_eq!(strategy, default);
        assert_eq!(scope, StrategyScope::Default);
    }

    #[test]
    fn explicit_strategy_override_still_wins_for_priority_group() {
        let registry = priority_registry();

        let (strategy, scope) = routing_context(
            &registry,
            &RoutingStrategy::UsageBasedV2,
            Some("least-busy"),
            &ResolveKind::RoutingGroup,
            "m-group",
        );

        assert_eq!(strategy, RoutingStrategy::LeastBusy);
        assert_eq!(scope, StrategyScope::Override);
    }

    #[rstest]
    #[case::honoured(Some("least-busy"), Some(RoutingStrategy::LeastBusy))]
    #[case::shuffle(Some("simple-shuffle"), Some(RoutingStrategy::SimpleShuffle))]
    #[case::lar1_ignored(Some("lar1"), None)]
    #[case::unknown_ignored(Some("bogus"), None)]
    #[case::unset(None, None)]
    fn only_overridable_strategies_override(
        #[case] raw: Option<&str>,
        #[case] expected: Option<RoutingStrategy>,
    ) {
        assert_eq!(request_override(raw), expected);
    }
}
