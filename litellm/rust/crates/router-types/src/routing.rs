//! How a router picks among a model group's deployments.

use std::collections::BTreeMap;

use serde::Deserialize;

/// `routing_strategy`, as `Router(routing_strategy=...)` takes it.
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
)]
pub enum RoutingStrategy {
    #[default]
    #[strum(serialize = "simple-shuffle")]
    SimpleShuffle,
    #[strum(serialize = "least-busy")]
    LeastBusy,
    #[strum(serialize = "usage-based-routing")]
    UsageBased,
    #[strum(serialize = "usage-based-routing-v2")]
    UsageBasedV2,
    #[strum(serialize = "latency-based-routing")]
    LatencyBased,
    #[strum(serialize = "cost-based-routing")]
    CostBased,
    #[strum(serialize = "provider-budget-routing")]
    ProviderBudget,
    #[strum(serialize = "lar1")]
    Lar1,
    /// A routing group whose members are tried in `model_priorities` order.
    #[strum(serialize = "priority")]
    Priority,
    /// `routing_strategy=None` passed explicitly (sent as `"None"`): stored as None, so no
    /// strategy arm matches and every pick finds no deployment.
    #[strum(serialize = "None")]
    Unset,
    #[strum(default, transparent)]
    Other(String),
}

/// `routing_groups`: a named subset of model groups sharing a routing strategy.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RoutingGroupSpec {
    pub group_name: String,
    pub models: Vec<String>,
    pub routing_strategy: RoutingStrategy,
    /// A priority group's member priorities, served as each member's `order`.
    #[serde(default)]
    pub model_priorities: Option<BTreeMap<String, i64>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::simple_shuffle("simple-shuffle", RoutingStrategy::SimpleShuffle)]
    #[case::least_busy("least-busy", RoutingStrategy::LeastBusy)]
    #[case::usage_based("usage-based-routing", RoutingStrategy::UsageBased)]
    #[case::usage_based_v2("usage-based-routing-v2", RoutingStrategy::UsageBasedV2)]
    #[case::latency_based("latency-based-routing", RoutingStrategy::LatencyBased)]
    #[case::cost_based("cost-based-routing", RoutingStrategy::CostBased)]
    #[case::provider_budget("provider-budget-routing", RoutingStrategy::ProviderBudget)]
    #[case::lar1("lar1", RoutingStrategy::Lar1)]
    #[case::priority("priority", RoutingStrategy::Priority)]
    #[case::unset("None", RoutingStrategy::Unset)]
    #[case::unknown("my-strategy", RoutingStrategy::Other("my-strategy".to_owned()))]
    fn routing_strategy_parses_its_wire_name(
        #[case] wire: &str,
        #[case] expected: RoutingStrategy,
    ) {
        assert_eq!(wire.parse::<RoutingStrategy>().unwrap(), expected);
        assert_eq!(expected.to_string(), wire);
    }
}
