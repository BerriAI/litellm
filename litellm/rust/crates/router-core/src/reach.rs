//! Where a request for a name can be served: the model group the name resolves to, every group
//! its fallbacks reach, and the host steps a call served by each one makes beyond its attempt.
//! Read from what the call loop reads, so a host can tell what a call will ask of it before the
//! call runs. Nothing here refuses or validates, and the router serves every name as it did.
//!
//! Only a name that is a model group, itself or through `model_group_alias`, is followed. The
//! call loop also resolves a deployment id, a routing group, a wildcard and the first default
//! fallback, and a fallback target only one of those serves is reported as not served.

use std::{collections::BTreeSet, sync::Arc};

use litellm_router_types::RoutingStrategy;

use crate::{
    Router,
    config::RouterDynamicConfig,
    fallback::{FallbackTarget, Fallbacks},
    registry::{Deployment, DeploymentFilter, Registry, ResolveKind},
    settings::Settings,
    strategy::{StrategyScope, implemented, routing_context},
};

/// One model group a request can be served by.
#[derive(Clone, Debug, PartialEq)]
pub struct Reached {
    /// The model group, after aliases.
    pub group: String,
    /// `false` for a fallback target that names no model group.
    pub served: bool,
    /// The host calls beyond `attempt` that a call served by this group makes.
    pub steps: Vec<HostStep>,
}

/// A call the router makes on its host or its cache because of a setting.
#[derive(Clone, Debug, PartialEq)]
pub enum HostStep {
    /// `RouterHost::select` with `SelectStage::Strategy`, for the selector the scope names.
    Pick(RoutingStrategy, StrategyScope),
    /// `RouterHost::pre_call_facts`.
    PreCallFacts,
    /// `RouterCache::health_states`.
    HealthStates,
    /// `RouterHost::refuses_fallbacks`, over a fallback written as a dict in the list this names.
    FallbackParams(&'static str),
}

impl HostStep {
    /// The router setting that causes the step, as the config spells it.
    pub fn setting(&self) -> &'static str {
        match self {
            Self::Pick(_, StrategyScope::Group(_)) => "routing_groups",
            Self::Pick(_, StrategyScope::Default | StrategyScope::Override) => "routing_strategy",
            Self::PreCallFacts => "enable_pre_call_checks",
            Self::HealthStates => "enable_health_check_routing",
            Self::FallbackParams(list) => list,
        }
    }
}

type Chain = (&'static str, Vec<FallbackTarget>);

/// The chain each fallback list holds for `lookup`, as `fallback_targets` in `call.rs` finds it,
/// under the setting that holds the list, then `default_fallbacks`.
fn chains(dynamic: &RouterDynamicConfig, lookup: &[&str]) -> Vec<Chain> {
    let keys = &dynamic.fallback_keys;
    let regular = if dynamic.fallbacks.is_client_side(keys) {
        Some(dynamic.fallbacks.as_chain())
    } else {
        dynamic.fallbacks.lookup_groups(lookup, keys)
    };
    let error_specific = |list: &Option<Fallbacks>| {
        list.as_ref()
            .and_then(|list| list.lookup_groups(lookup, keys))
    };
    let defaults = dynamic
        .default_fallbacks
        .iter()
        .cloned()
        .map(FallbackTarget::Name)
        .collect();
    [
        ("fallbacks", regular),
        (
            "context_window_fallbacks",
            error_specific(&dynamic.context_window_fallbacks),
        ),
        (
            "content_policy_fallbacks",
            error_specific(&dynamic.content_policy_fallbacks),
        ),
        ("default_fallbacks", Some(defaults)),
    ]
    .into_iter()
    .filter_map(|(setting, chain)| Some((setting, chain?)))
    .collect()
}

/// The steps of a level serving `group`: what its pick asks when a deployment serves it, then
/// what its fallback `chains` ask.
fn steps(
    registry: &Registry,
    settings: &Settings,
    group: &str,
    served: bool,
    chains: &[Chain],
) -> Vec<HostStep> {
    let (strategy, scope) = routing_context(
        registry,
        &settings.routing_strategy,
        None,
        &ResolveKind::Group,
        group,
    );
    let pick = [
        (!implemented(&strategy)).then_some(HostStep::Pick(strategy, scope)),
        settings
            .enable_pre_call_checks
            .then_some(HostStep::PreCallFacts),
        settings
            .enable_health_check_routing
            .then_some(HostStep::HealthStates),
    ];
    let params = chains
        .iter()
        .filter(|(_, chain)| chain.iter().any(|target| target.params().is_some()))
        .map(|(setting, _)| HostStep::FallbackParams(setting));
    pick.into_iter()
        .flatten()
        .filter(|_| served)
        .chain(params)
        .collect()
}

/// `_direct_fallback_model_names` over every fallback type: the groups each list names as
/// `model`'s chain. Every chain routing can run is followed, where Python follows fewer: a list
/// of bare names is the whole chain, not its last name, and a chain of every group (`*`) is
/// followed beside `default_fallbacks`, which Python reads over it.
fn direct_fallback_model_names(dynamic: &RouterDynamicConfig, model: &str) -> Vec<String> {
    let keys = &dynamic.fallback_keys;
    let error_specific = [
        &dynamic.context_window_fallbacks,
        &dynamic.content_policy_fallbacks,
    ]
    .into_iter()
    .flatten()
    .filter_map(|list| list.lookup_groups(&[model], keys));
    let regular = if dynamic.fallbacks.is_client_side(keys) {
        vec![dynamic.fallbacks.as_chain()]
    } else {
        [
            dynamic.fallbacks.lookup_groups(&[model], keys),
            dynamic
                .fallbacks
                .chain_with_defaults(model, keys, &dynamic.default_fallbacks),
        ]
        .into_iter()
        .flatten()
        .collect()
    };
    regular
        .into_iter()
        .chain(error_specific)
        .flatten()
        .filter_map(|target| target.model().map(str::to_owned))
        .filter(|name| !name.is_empty())
        .collect()
}

impl Router {
    /// `router_fallback_model_names` over every fallback type: each group a failure on `model`
    /// can reach through the router's fallback lists, chained and generic (`*`), with
    /// `default_fallbacks` read as the Python router folds them in. `model` itself is left out,
    /// and names are taken as written, aliases unresolved.
    pub fn fallback_model_names(&self, model: &str) -> BTreeSet<String> {
        fallback_model_names(&self.core.snapshot.load().settings.dynamic, model)
    }

    /// Python's `_opt_in_model_groups`: the deployments of every model group a request for
    /// `requested` can be served by. Each name goes through its alias and the router's fallbacks,
    /// then each found deployment's own group joins, until no new group turns up. Each new group's
    /// fallbacks are followed too, where Python follows only those of a deployment's group, so a
    /// fallback reached through an alias is judged as the router would serve it. A group that
    /// no deployment serves is an empty one. `team_id` picks a team's own deployments of a group
    /// when it has any. Wildcard deployments and a provider's group, which Python also reads, are
    /// not followed yet.
    pub fn opt_in_groups(
        &self,
        requested: &BTreeSet<String>,
        team_id: Option<&str>,
    ) -> Vec<Vec<Arc<Deployment>>> {
        let snapshot = self.core.snapshot.load();
        let registry = &snapshot.registry;
        let dynamic = &snapshot.settings.dynamic;
        let alias = |name: &str| registry.alias_of(name).unwrap_or(name).to_owned();
        let fallbacks = |names: &BTreeSet<String>| -> BTreeSet<String> {
            names
                .iter()
                .flat_map(|name| fallback_model_names(dynamic, name))
                .map(|name| alias(&name))
                .collect()
        };
        let routed: BTreeSet<String> = requested.iter().map(|name| alias(name)).collect();
        let groups: BTreeSet<String> = routed
            .iter()
            .cloned()
            .chain(fallbacks(&requested.union(&routed).cloned().collect()))
            .collect();
        let mut every = Vec::new();
        let mut reached = groups.clone();
        let mut frontier = groups;
        while !frontier.is_empty() {
            let deployments: Vec<Vec<Arc<Deployment>>> = frontier
                .iter()
                .map(|group| group_deployments(registry, group, team_id))
                .collect();
            let routed_groups: BTreeSet<String> = deployments
                .iter()
                .flatten()
                .map(|deployment| deployment.model_name.clone())
                .filter(|name| !reached.contains(name))
                .collect();
            every.extend(deployments);
            let next: BTreeSet<String> = routed_groups
                .iter()
                .cloned()
                .chain(fallbacks(
                    &frontier.union(&routed_groups).cloned().collect(),
                ))
                .filter(|name| !reached.contains(name))
                .collect();
            reached.extend(next.iter().cloned());
            frontier = next;
        }
        every
    }
}

/// A team's own deployments of `group` when it has any, else the group's.
fn group_deployments(
    registry: &Registry,
    group: &str,
    team_id: Option<&str>,
) -> Vec<Arc<Deployment>> {
    let team = team_id
        .map(|team_id| {
            registry.deployments(&DeploymentFilter {
                model_name: Some(group),
                team_id: Some(team_id),
                provider: None,
            })
        })
        .filter(|deployments| !deployments.is_empty());
    team.unwrap_or_else(|| registry.group(group).unwrap_or_default())
}

fn fallback_model_names(dynamic: &RouterDynamicConfig, model: &str) -> BTreeSet<String> {
    let mut reached = BTreeSet::from([model.to_owned()]);
    let mut frontier = vec![model.to_owned()];
    while !frontier.is_empty() {
        let next: BTreeSet<String> = frontier
            .iter()
            .flat_map(|name| direct_fallback_model_names(dynamic, name))
            .filter(|name| !reached.contains(name))
            .collect();
        reached.extend(next.iter().cloned());
        frontier = next.into_iter().collect();
    }
    reached.remove(model);
    reached
}

impl Router {
    /// The group `name` resolves to, then every group its fallbacks reach, in the order reached.
    /// `None` when `name` is no model group, itself or through an alias.
    pub fn reach(&self, name: &str) -> Option<Vec<Reached>> {
        let snapshot = self.core.snapshot.load();
        let registry = &snapshot.registry;
        let serves = |group: &str| {
            registry
                .group(group)
                .is_some_and(|deployments| !deployments.is_empty())
        };
        if !serves(registry.alias_of(name).unwrap_or(name)) {
            return None;
        }
        let mut names = vec![name.to_owned()];
        let mut reached: Vec<Reached> = Vec::new();
        let mut next = 0;
        while let Some(written) = names.get(next).cloned() {
            next += 1;
            let chains = chains(&snapshot.settings.dynamic, &[&written, name]);
            let group = registry.alias_of(&written).unwrap_or(&written);
            let served = serves(group);
            let steps = steps(registry, &snapshot.settings, group, served, &chains);
            match reached.iter_mut().find(|seen| seen.group == group) {
                Some(seen) => {
                    for step in steps {
                        if !seen.steps.contains(&step) {
                            seen.steps.push(step);
                        }
                    }
                }
                None => reached.push(Reached {
                    group: group.to_owned(),
                    served,
                    steps,
                }),
            }
            for target in chains.iter().flat_map(|(_, chain)| chain) {
                if let Some(target) = target.model()
                    && !names.iter().any(|seen| seen == target)
                {
                    names.push(target.to_owned());
                }
            }
        }
        Some(reached)
    }
}
