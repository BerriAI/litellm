//! The router settings router-core serves only through a host step the native host does not
//! answer. router-core asks for such a step only on the calls the setting reaches, and every
//! request one of them reaches goes to Python, so the native router is built from the settings as
//! written and never asks.

use std::collections::BTreeMap;

use litellm_config::{RouterSettings, RoutingGroupSpec, RoutingStrategy};
use litellm_router_core::strategy::implemented;

/// Whether router-core asks the host to pick a deployment by `strategy`.
pub(super) fn picked_by_the_host(strategy: &Option<RoutingStrategy>) -> bool {
    strategy
        .as_ref()
        .is_some_and(|strategy| !implemented(strategy))
}

fn pins(affinity: &Option<BTreeMap<String, Vec<String>>>, group: &str) -> bool {
    affinity
        .as_ref()
        .is_some_and(|groups| groups.contains_key(group))
}

/// A routing group with its own strategy picks for each model group it holds. Its own name is no
/// model group, so `model` leaves a request for it to Python first.
fn picks_for(routing_groups: &Option<Vec<RoutingGroupSpec>>, group: &str) -> bool {
    routing_groups.iter().flatten().any(|routing| {
        !implemented(&routing.routing_strategy) && routing.models.iter().any(|model| model == group)
    })
}

/// The setting that makes router-core ask the native host for a step on a call for `group`.
pub(super) fn group(settings: &RouterSettings, group: &str) -> Option<Box<str>> {
    let RouterSettings {
        model_group_affinity_config,
        routing_groups,
        ..
    } = settings;
    set!(
        "router_settings.",
        [],
        by: [
            model_group_affinity_config => |affinity| pins(affinity, group),
            routing_groups => |routing_groups| picks_for(routing_groups, group),
        ]
    )
    .next()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::simple_shuffle("simple-shuffle", false)]
    #[case::none_written_out("None", false)]
    #[case::another_strategy("least-busy", true)]
    fn only_simple_shuffle_is_picked_without_the_host(#[case] written: &str, #[case] host: bool) {
        assert_eq!(picked_by_the_host(&written.parse().ok()), host);
    }
}
