//! A requested name to the deployments that may serve this caller: a port of
//! `_common_checks_available_deployment`. In order: a specific deployment, a deployment id, the
//! alias, a routing group, then for a name no deployment has, a team's public name, every team's
//! public name for a proxy admin, a wildcard; else the group (team-first), kept to the caller's
//! access groups, then deployments whose litellm model is the name, then the first default
//! fallback. Every arm drops strategy markers and withholds deployments reserved for other teams.

use std::{collections::BTreeSet, sync::Arc};

use jiff::Timestamp;

use serde_json::Value;

use crate::{
    access::{filter_by_access_groups, filter_reserved},
    error::Error,
    registry::{Deployment, DeploymentSpec, Registry, Resolution, ResolveKind},
    select::is_strategy_marker,
};

/// Who is asking, as the request's auth metadata says.
#[derive(Clone, Copy, Debug, Default)]
pub struct Caller<'a> {
    /// `user_api_key_team_id`.
    pub team_id: Option<&'a str>,
    /// The caller's `user_role` is `proxy_admin`.
    pub proxy_admin: bool,
    /// The caller's `models` plus `team_models`.
    pub allowed_models: &'a [String],
}

/// `_core_should_include_deployment`: the team's own deployment under its public name, or a
/// deployment named `model_name` the team may use.
fn included(deployment: &Deployment, model_name: &str, team_id: Option<&str>) -> bool {
    let info = &deployment.model_info;
    if team_id.is_some()
        && info.team_id.as_deref() == team_id
        && info.team_public_model_name.as_deref() == Some(model_name)
    {
        return true;
    }
    deployment.model_name == model_name
        && (team_id.is_none() || info.team_id.is_none() || info.team_id.as_deref() == team_id)
}

/// `_core_get_all_deployments(model_name, team_id)`: the team's deployments under that public
/// name when it has any, else the deployments named `model_name` the team may use; a name the
/// model-name index does not hold is looked for in every deployment, for a team.
pub fn all_deployments(
    registry: &Registry,
    model_name: &str,
    team_id: Option<&str>,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let keep = |deployments: Vec<Arc<Deployment>>| -> Vec<Arc<Deployment>> {
        deployments
            .into_iter()
            .filter(|d| included(d, model_name, team_id))
            .collect()
    };
    // A written index position past the model list raises `IndexError` where it is read.
    let out_of_range = || Error::IndexOutOfRange {
        model: model_name.to_owned(),
    };
    if let Some(team) = team_id {
        if registry.team_index_overflows(team, model_name) {
            return Err(out_of_range());
        }
        if let Some(own) = registry.team_named(team, model_name) {
            let own = keep(own);
            if !own.is_empty() {
                return Ok(own);
            }
        }
    }
    if registry.name_index_overflows(model_name) {
        return Err(out_of_range());
    }
    Ok(match registry.named(model_name) {
        Some(named) => keep(named),
        None if team_id.is_some() => keep(registry.all().to_vec()),
        None => Vec::new(),
    })
}

/// `_team_deployments_across_teams`: every team's deployments under public name `model`, for a
/// proxy admin calling without a team; a name several teams share is refused.
fn across_teams(registry: &Registry, model: &str) -> Result<Vec<Arc<Deployment>>, Error> {
    if registry.across_teams_index_overflows(model) {
        return Err(Error::IndexOutOfRange {
            model: model.to_owned(),
        });
    }
    let deployments = registry.team_named_across_teams(model);
    let teams: BTreeSet<&str> = deployments
        .iter()
        .filter_map(|d| d.model_info.team_id.as_deref())
        .collect();
    if teams.len() > 1 {
        return Err(Error::MultipleTeams {
            model: model.to_owned(),
        });
    }
    Ok(deployments)
}

/// The access groups of the rows `_core_get_model_list(model_name, team_id)` lists: the
/// group's deployments, an alias's target deployments, a routing group's rows (which carry no
/// access groups), and only when there are none of those, the wildcard's deployments.
fn model_access_groups(
    registry: &Registry,
    model: &str,
    team_id: Option<&str>,
) -> BTreeSet<String> {
    let mut rows = all_deployments(registry, model, team_id).unwrap_or_default();
    let mut stripped_rows = 0;
    if let Some(target) = registry.alias_of(model) {
        match registry.routing_group_named(target) {
            Some(group) => stripped_rows += routing_group_rows(registry, group, None).len(),
            None => rows.extend(all_deployments(registry, target, None).unwrap_or_default()),
        }
    }
    if let Some(group) = registry.routing_group_named(model) {
        stripped_rows += routing_group_rows(registry, group, None).len();
    }
    if rows.is_empty() && stripped_rows == 0 {
        rows = registry
            .wildcard_for(model)
            .map(|(_, deployments)| deployments)
            .unwrap_or_default();
    }
    rows.iter()
        .flat_map(|d| d.model_info.access_groups.iter().cloned())
        .collect()
}

fn routing_group_rows(
    registry: &Registry,
    group: &crate::registry::RoutingGroupSpec,
    team_id: Option<&str>,
) -> Vec<Arc<Deployment>> {
    crate::routing_groups::members(group, |member| {
        all_deployments(registry, member, team_id).unwrap_or_default()
    })
}

/// `_filter_reserved_deployments` for deployments resolved as `model`: markers dropped (only
/// markers is refused), then the deployments an active access window withholds; when that
/// leaves nothing, the window is named in the refusal.
fn reserved(
    model: &str,
    deployments: Vec<Arc<Deployment>>,
    team_id: Option<&str>,
    now: Timestamp,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let any = !deployments.is_empty();
    let selectable: Vec<Arc<Deployment>> = deployments
        .into_iter()
        .filter(|d| !is_strategy_marker(d))
        .collect();
    if any && selectable.is_empty() {
        return Err(Error::OnlyMarkers {
            model: model.to_owned(),
        });
    }
    let (kept, blocking) = filter_reserved(selectable, team_id, now);
    match blocking {
        Some(window) if kept.is_empty() => Err(Error::Reserved {
            model: model.to_owned(),
            until: window.end_hhmm(),
            timezone: window.timezone,
        }),
        _ => Ok(kept),
    }
}

/// The first default fallback a name no deployment serves is routed to.
pub fn default_target(dynamic: &crate::config::RouterDynamicConfig) -> Option<&str> {
    dynamic
        .fallbacks
        .default_target()
        .or_else(|| dynamic.default_fallbacks.first().map(String::as_str))
}

/// The router's `default_deployment`, as resolution serves it.
#[derive(Clone, Copy, Debug)]
pub enum DefaultDeployment<'a> {
    Spec(&'a DeploymentSpec),
    /// A value Python cannot copy for a name: resolving to it raises.
    Raises,
}

/// The default deployment as served for `model`: a copy whose `litellm_params.model` is the name.
fn default_for(spec: &DeploymentSpec, model: &str) -> Option<Arc<Deployment>> {
    let mut spec = spec.clone();
    spec.litellm_params
        .insert("model".to_owned(), Value::String(model.to_owned()));
    Deployment::from_spec(spec).ok().map(Arc::new)
}

/// `_core_get_routable_deployments`: the deployments resolution would consider for `request`,
/// before any team, cooldown or strategy filter; nothing when it resolves to nothing or refuses.
pub fn routable(
    registry: &Registry,
    dynamic: &crate::config::RouterDynamicConfig,
    request: &crate::host::RouterRequest,
) -> Vec<Arc<Deployment>> {
    let caller = Caller {
        team_id: request.team_id.as_deref(),
        proxy_admin: request.proxy_admin,
        allowed_models: &request.allowed_models,
    };
    match resolve_request(
        registry,
        &request.model,
        false,
        caller,
        default_target(dynamic),
        dynamic.default_ref(),
        Timestamp::now(),
    ) {
        Ok(Some(resolution)) => resolution.deployments,
        Ok(None) | Err(_) => Vec::new(),
    }
}

/// Resolve `name` for `caller`. `None` when nothing serves it (`no_healthy_deployments`).
pub fn resolve_request(
    registry: &Registry,
    name: &str,
    specific_deployment: bool,
    caller: Caller<'_>,
    default_target: Option<&str>,
    default_deployment: Option<DefaultDeployment<'_>>,
    now: Timestamp,
) -> Result<Option<Resolution>, Error> {
    let team_id = caller.team_id;
    let resolved = |kind, group: &str, deployments| {
        Ok(Some(Resolution {
            kind,
            group: group.to_owned(),
            deployments,
        }))
    };
    if specific_deployment {
        let matching = registry
            .all()
            .iter()
            .filter(|d| d.params.model == name)
            .cloned()
            .collect();
        let kept = reserved(name, matching, team_id, now)?;
        return resolved(ResolveKind::SpecificDeployment, name, kept);
    }
    if !registry.has_name(name)
        && let Some(deployment) = registry.deployment(name)
    {
        let group = deployment.model_name.clone();
        let litellm_model = deployment.params.model.clone();
        let kept = reserved(&litellm_model, vec![deployment], team_id, now)?;
        return resolved(ResolveKind::Id, &group, kept);
    }
    let aliased = registry.alias_of(name);
    let model = aliased.unwrap_or(name);
    let routing_group = registry
        .routing_group_named(model)
        .map(|group| routing_group_rows(registry, group, team_id));
    if routing_group.is_none() && !registry.has_name(model) {
        let early = match team_id {
            Some(team) => Some(all_deployments(registry, model, Some(team))?),
            None if caller.proxy_admin => Some(across_teams(registry, model)?),
            None => None,
        }
        .filter(|deployments| !deployments.is_empty());
        if let Some(deployments) = early {
            let kept = reserved(model, deployments, team_id, now)?;
            return resolved(ResolveKind::Group, model, kept);
        }
        if let Some((pattern, deployments)) = registry.wildcard_for(model) {
            let kept = reserved(model, deployments, team_id, now)?;
            return resolved(ResolveKind::Wildcard, &pattern, kept);
        }
        match default_deployment {
            Some(DefaultDeployment::Raises) => {
                return Err(Error::DefaultDeploymentRaises {
                    model: model.to_owned(),
                });
            }
            Some(DefaultDeployment::Spec(spec)) => {
                if let Some(deployment) = default_for(spec, model) {
                    let kept = reserved(model, vec![deployment], team_id, now)?;
                    return resolved(ResolveKind::Default, model, kept);
                }
            }
            None => {}
        }
    }
    let (kind, listed) = match routing_group {
        Some(deployments) => (ResolveKind::RoutingGroup, deployments),
        None if aliased.is_some() => (
            ResolveKind::Alias,
            all_deployments(registry, model, team_id)?,
        ),
        None => (
            ResolveKind::Group,
            all_deployments(registry, model, team_id)?,
        ),
    };
    let narrow = |model: &str, deployments: Vec<Arc<Deployment>>| {
        let allowed = filter_by_access_groups(model, deployments, caller.allowed_models, || {
            model_access_groups(registry, model, team_id)
        });
        reserved(model, allowed, team_id, now)
    };
    let listed_any = !listed.is_empty();
    let deployments = narrow(model, listed)?;
    if !deployments.is_empty() {
        return resolved(kind, model, deployments);
    }
    let mut access_emptied = listed_any;
    if !listed_any {
        // The caller may have named a deployment's litellm model instead.
        let by_model: Vec<Arc<Deployment>> = registry
            .all()
            .iter()
            .filter(|d| d.params.model == model)
            .cloned()
            .collect();
        let found = !by_model.is_empty();
        let deployments = narrow(model, by_model)?;
        if !deployments.is_empty() {
            return resolved(ResolveKind::Group, model, deployments);
        }
        access_emptied = found;
    }
    if let Some(target) = default_target.filter(|_| !access_emptied) {
        let deployments = narrow(target, all_deployments(registry, target, team_id)?)?;
        if !deployments.is_empty() {
            return resolved(ResolveKind::Group, target, deployments);
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{Value, json};

    use super::*;
    use crate::registry::DeploymentSpec;

    fn registry(entries: Value) -> Registry {
        let specs: Vec<DeploymentSpec> = serde_json::from_value(entries).unwrap();
        Registry::build(specs, BTreeMap::new(), Vec::new()).unwrap()
    }

    fn ids(resolution: Option<Resolution>) -> Vec<String> {
        resolution
            .map(|r| r.deployments.iter().map(|d| d.id.clone()).collect())
            .unwrap_or_default()
    }

    #[test]
    fn an_access_group_only_caller_is_kept_to_its_groups() {
        let registry = registry(json!([
            {"model_name": "g", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a", "access_groups": ["A"]}},
            {"model_name": "g", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "b", "access_groups": ["B"]}},
        ]));
        let allowed = vec!["A".to_owned()];
        let caller = Caller {
            allowed_models: &allowed,
            ..Caller::default()
        };
        let resolved = resolve_request(&registry, "g", false, caller, None, None, Timestamp::now());
        assert_eq!(ids(resolved.unwrap()), ["a"]);
        // The litellm model string is not narrowed (python-fixes-to-make.md).
        let by_model = resolve_request(
            &registry,
            "openai/b",
            false,
            caller,
            None,
            None,
            Timestamp::now(),
        );
        assert_eq!(ids(by_model.unwrap()), ["b"]);
    }

    #[test]
    fn a_proxy_admin_is_refused_a_public_name_several_teams_share() {
        let registry = registry(json!([
            {"model_name": "x", "litellm_params": {"model": "openai/x"}, "model_info": {"id": "x", "team_id": "tx", "team_public_model_name": "pub"}},
            {"model_name": "y", "litellm_params": {"model": "openai/y"}, "model_info": {"id": "y", "team_id": "ty", "team_public_model_name": "pub"}},
        ]));
        let caller = Caller {
            proxy_admin: true,
            ..Caller::default()
        };
        let refused = resolve_request(
            &registry,
            "pub",
            false,
            caller,
            None,
            None,
            Timestamp::now(),
        );
        assert!(matches!(refused, Err(Error::MultipleTeams { model }) if model == "pub"));
    }
}
