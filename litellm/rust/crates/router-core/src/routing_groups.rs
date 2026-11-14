//! Routing groups: a callable name over the deployments of its member model groups
//! (`_get_routing_group_deployments`). A priority group serves each member's priority as the
//! member deployments' `order` (`apply_routing_group_priority`), so the order filter and order
//! fallbacks walk the priorities.

use std::sync::Arc;

use serde_json::Value;

use crate::registry::{Deployment, RoutingGroupSpec};

/// The member deployments of `group`, in member order, with a priority group's priorities as
/// their `order`.
pub fn members(
    group: &RoutingGroupSpec,
    deployments_of: impl Fn(&str) -> Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    let priorities = group
        .model_priorities
        .as_ref()
        .filter(|_| is_priority(group));
    group
        .models
        .iter()
        .flat_map(|member| {
            let priority = priorities.and_then(|p| p.get(member)).copied();
            deployments_of(member)
                .into_iter()
                .map(move |deployment| match priority {
                    Some(order) => with_order(&deployment, order),
                    None => deployment,
                })
        })
        .collect()
}

/// Whether `group` is a priority group.
pub fn is_priority(group: &RoutingGroupSpec) -> bool {
    matches!(&group.routing_strategy, crate::config::RoutingStrategy::Other(name) if name == "priority")
}

fn with_order(deployment: &Arc<Deployment>, order: i64) -> Arc<Deployment> {
    let mut prioritized = Deployment::clone(deployment);
    prioritized
        .raw_params
        .insert("order".to_owned(), Value::from(order));
    Arc::new(prioritized)
}
