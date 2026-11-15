//! A requested name to the deployments that may serve this caller: a port of
//! `_common_checks_available_deployment`. In order: a specific deployment, a deployment id, the
//! alias, a routing group, then for a name no deployment has, a team's public name, every team's
//! public name for a proxy admin, a wildcard; else the group (team-first), kept to the caller's
//! access groups, then deployments whose litellm model is the name, then the first default
//! fallback. Every arm drops strategy markers and deployments that refuse the request's endpoint,
//! and withholds deployments reserved for other teams.

use std::{collections::BTreeSet, sync::Arc};

use jiff::Timestamp;

use serde_json::Value;

use crate::{
    access::{filter_by_access_groups, filter_reserved},
    eligibility::eligible_deployments,
    error::Error,
    host_index::{pattern_to_regex, regex_match},
    registry::{Deployment, DeploymentSpec, Registry, Resolution, ResolveKind, routed_deployment},
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
    /// The endpoint the pick runs under (`endpoint_selection`), which deployments may refuse.
    pub endpoint: Option<&'a str>,
    /// The model `endpoint_selection` was given: the name an endpoint refusal reads.
    pub requested: &'a str,
    /// `f"{provider}/{model}"`, the spelling `PatternMatchRouter.get_pattern` tries when no
    /// pattern matches the name itself.
    pub wildcard_request: Option<&'a str>,
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
/// markers is refused), those that refuse the caller's endpoint dropped (`_drop_strategy_markers`),
/// then the deployments an active access window withholds; when that leaves nothing, the window
/// is named in the refusal.
fn reserved(
    model: &str,
    deployments: Vec<Arc<Deployment>>,
    caller: Caller<'_>,
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
    let selectable = eligible_deployments(model, caller, selectable)?;
    let (kept, blocking) = filter_reserved(selectable, caller.team_id, now);
    match blocking {
        Some(window) if kept.is_empty() => Err(Error::Reserved {
            model: model.to_owned(),
            until: window.end_hhmm(),
            timezone: window.timezone,
        }),
        _ => Ok(kept),
    }
}

/// `PatternMatchRouter.route`: copies of a pattern's deployments serving `request`.
fn routed_wildcard(
    pattern: &str,
    request: &str,
    deployments: &[Arc<Deployment>],
) -> Vec<Arc<Deployment>> {
    let groups = regex_match(&pattern_to_regex(pattern), request).unwrap_or_default();
    deployments
        .iter()
        .map(|deployment| routed_deployment(deployment, &groups, request))
        .collect()
}

/// `get_pattern` for `model`: the most specific pattern matching it as spelled, else its
/// provider-prefixed spelling, with the pattern's deployments serving that request.
fn wildcard_route(
    registry: &Registry,
    model: &str,
    wildcard_request: Option<&str>,
) -> Option<(String, Vec<Arc<Deployment>>)> {
    let route = |request: &str| {
        registry
            .wildcard_for(request)
            .map(|(pattern, deployments)| {
                let routed = routed_wildcard(&pattern, request, &deployments);
                (pattern, routed)
            })
    };
    route(model).or_else(|| route(wildcard_request?))
}

/// The name resolution would look for under its provider (`get_pattern`'s second spelling), which
/// only the host can infer: one that reaches the pattern routers and that no pattern matches as
/// spelled.
pub(crate) fn wildcard_name<'a>(
    registry: &'a Registry,
    name: &'a str,
    specific_deployment: bool,
    team_id: Option<&str>,
) -> Option<&'a str> {
    let model = registry.alias_of(name).unwrap_or(name);
    let lookup = crate::lookup::Lookup::new(registry);
    let team_patterns = team_id.is_some_and(|team| lookup.has_team_wildcards(team));
    let team_misses = || team_id.is_none_or(|team| lookup.team_wildcard(model, team).is_empty());
    let global_misses = || registry.has_wildcards() && registry.wildcard_for(model).is_none();
    ((team_patterns || registry.has_wildcards())
        && !specific_deployment
        && !registry.has_name(model)
        && (registry.has_name(name) || registry.deployment(name).is_none())
        && registry.routing_group_named(model).is_none()
        && team_misses()
        && (team_patterns || global_misses()))
    .then_some(model)
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
        endpoint: request.endpoint.as_deref(),
        requested: &request.model,
        wildcard_request: request.wildcard_request.as_deref(),
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
        let kept = reserved(name, matching, caller, now)?;
        return resolved(ResolveKind::SpecificDeployment, name, kept);
    }
    if !registry.has_name(name)
        && let Some(deployment) = registry.deployment(name)
    {
        let group = deployment.model_name.clone();
        let litellm_model = deployment.params.model.clone();
        let accessible = if caller.allowed_models.iter().any(|m| m == name) {
            vec![deployment]
        } else {
            filter_by_access_groups(&group, vec![deployment], caller.allowed_models, || {
                model_access_groups(registry, &group, team_id)
            })
        };
        if accessible.is_empty() {
            return Ok(None);
        }
        let kept = reserved(&litellm_model, accessible, caller, now)?;
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
            let kept = reserved(model, deployments, caller, now)?;
            return resolved(ResolveKind::Group, model, kept);
        }
        // `_core_wildcard_route_deployments`: the team's own pattern, else the proxy-wide one,
        // each tried as spelled and then under the name's provider (`get_pattern`).
        let team_owned = team_id.and_then(|team| {
            let lookup = crate::lookup::Lookup::new(registry);
            [Some(model), caller.wildcard_request]
                .into_iter()
                .flatten()
                .map(|request| (request, lookup.team_wildcard(request, team)))
                .find(|(_, deployments)| !deployments.is_empty())
        });
        if let Some((request, deployments)) = team_owned {
            let routed = deployments.iter().flat_map(|deployment| {
                let public = deployment.model_info.team_public_model_name.as_deref();
                let pattern = public.unwrap_or(request);
                routed_wildcard(pattern, request, std::slice::from_ref(deployment))
            });
            let kept = reserved(model, routed.collect(), caller, now)?;
            return resolved(ResolveKind::Wildcard, model, kept);
        }
        if let Some((pattern, deployments)) =
            wildcard_route(registry, model, caller.wildcard_request)
        {
            let kept = reserved(model, deployments, caller, now)?;
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
                    let kept = reserved(model, vec![deployment], caller, now)?;
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
        reserved(model, allowed, caller, now)
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
    if let Some(target) = default_target.filter(|target| !access_emptied && !target.is_empty()) {
        let deployments = narrow(target, all_deployments(registry, target, team_id)?)?;
        if !deployments.is_empty() {
            return resolved(ResolveKind::Group, target, deployments);
        }
        // `_common_checks_available_deployment` re-assigned the name: its error names the target.
        return Err(Error::UnknownModel {
            model: target.to_owned(),
        });
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::{fixture, rstest};
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

    #[fixture]
    fn shared_group() -> Registry {
        registry(json!([
            {"model_name": "shared", "litellm_params": {"model": "openai/pub"}, "model_info": {"id": "pub", "access_groups": ["public"]}},
            {"model_name": "shared", "litellm_params": {"model": "openai/priv"}, "model_info": {"id": "priv", "access_groups": ["private"]}},
        ]))
    }

    #[rstest]
    #[case::access_group_pins_its_own_deployment(&["public"], "pub", &["pub"])]
    #[case::access_group_cannot_pin_another_groups_deployment(&["public"], "priv", &[])]
    #[case::the_other_access_group_pins_its_deployment(&["private"], "priv", &["priv"])]
    #[case::the_owning_group_pins_any_of_its_deployments(&["shared"], "priv", &["priv"])]
    #[case::an_exact_id_grant_beside_another_access_group(&["public", "priv"], "priv", &["priv"])]
    #[case::an_unrestricted_caller_pins_any_deployment(&[], "priv", &["priv"])]
    #[case::every_model_pins_any_deployment(&["*"], "priv", &["priv"])]
    #[case::every_proxy_model_pins_any_deployment(&["all-proxy-models"], "priv", &["priv"])]
    #[case::the_group_name_stays_kept_to_the_callers_access_group(&["public"], "shared", &["pub"])]
    fn pinned_deployment_respects_caller_access_groups(
        shared_group: Registry,
        #[case] allowed: &[&str],
        #[case] name: &str,
        #[case] expected: &[&str],
    ) {
        let allowed_models: Vec<String> = allowed.iter().map(|m| (*m).to_owned()).collect();
        let caller = Caller {
            allowed_models: &allowed_models,
            ..Caller::default()
        };
        let resolved = resolve_request(
            &shared_group,
            name,
            false,
            caller,
            None,
            None,
            Timestamp::UNIX_EPOCH,
        );
        assert_eq!(ids(resolved.unwrap()), expected);
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

    fn wildcard_group() -> Value {
        json!([{
            "model_name": "openai/*",
            "litellm_params": {"model": "openai/*"},
            "model_info": {"id": "w", "supports_prompt_cache_breakpoint": true},
        }])
    }

    #[test]
    fn a_wildcard_serves_the_matched_model_without_its_cache_flags() {
        let registry = registry(wildcard_group());
        let resolved = resolve_request(
            &registry,
            "openai/gpt-4o",
            false,
            Caller::default(),
            None,
            None,
            Timestamp::now(),
        );
        let resolution = resolved.unwrap().unwrap();
        assert_eq!(resolution.kind, ResolveKind::Wildcard);
        assert_eq!(resolution.deployments[0].params.model, "openai/gpt-4o");
        assert_eq!(
            resolution.deployments[0].record()["model_info"],
            json!({"id": "w"})
        );
    }

    #[test]
    fn an_alias_to_a_wildcard_fills_the_model_from_its_target() {
        let specs: Vec<DeploymentSpec> = serde_json::from_value(wildcard_group()).unwrap();
        let aliases = BTreeMap::from([("al".to_owned(), "openai/gpt-4o".to_owned())]);
        let registry = Registry::build(specs, aliases, Vec::new()).unwrap();
        let resolved = resolve_request(
            &registry,
            "al",
            false,
            Caller::default(),
            None,
            None,
            Timestamp::now(),
        );
        assert_eq!(
            resolved.unwrap().unwrap().deployments[0].params.model,
            "openai/gpt-4o"
        );
    }

    #[test]
    fn a_name_no_pattern_matches_is_tried_under_its_provider() {
        let registry = registry(wildcard_group());
        assert_eq!(
            wildcard_name(&registry, "gpt-4o", false, None),
            Some("gpt-4o")
        );
        assert_eq!(wildcard_name(&registry, "openai/gpt-4o", false, None), None);
        let caller = Caller {
            wildcard_request: Some("openai/gpt-4o"),
            ..Caller::default()
        };
        let resolved = resolve_request(
            &registry,
            "gpt-4o",
            false,
            caller,
            None,
            None,
            Timestamp::now(),
        );
        let resolution = resolved.unwrap().unwrap();
        assert_eq!(resolution.group, "openai/*");
        assert_eq!(resolution.deployments[0].params.model, "openai/gpt-4o");
    }

    #[rstest]
    fn a_bare_name_is_tried_under_its_provider_against_the_teams_own_wildcard() {
        let registry = registry(json!([
            {"model_name": "model_name_tw_1", "litellm_params": {"model": "openai/*"},
             "model_info": {"id": "tw", "team_id": "TW", "team_public_model_name": "openai/*"}},
            {"model_name": "g", "litellm_params": {"model": "openai/g"}, "model_info": {"id": "g"}},
        ]));
        // No proxy-wide wildcard: only the team's patterns call for the provider spelling.
        assert_eq!(
            wildcard_name(&registry, "gpt-4o-mini", false, Some("TW")),
            Some("gpt-4o-mini")
        );
        assert_eq!(wildcard_name(&registry, "gpt-4o-mini", false, None), None);
        assert_eq!(
            wildcard_name(&registry, "openai/gpt-4o-mini", false, Some("TW")),
            None
        );
        let resolve = |team_id| {
            let caller = Caller {
                team_id,
                wildcard_request: Some("openai/gpt-4o-mini"),
                ..Caller::default()
            };
            resolve_request(
                &registry,
                "gpt-4o-mini",
                false,
                caller,
                None,
                None,
                Timestamp::now(),
            )
            .unwrap()
        };
        let resolution = resolve(Some("TW")).unwrap();
        assert_eq!(resolution.kind, ResolveKind::Wildcard);
        assert_eq!(resolution.group, "gpt-4o-mini");
        assert_eq!(ids(Some(resolution.clone())), ["tw"]);
        assert_eq!(resolution.deployments[0].params.model, "openai/gpt-4o-mini");
        assert!(resolve(Some("OTHER")).is_none());
    }

    #[test]
    fn a_deployment_named_by_id_refuses_the_endpoint_like_any_other() {
        let registry = registry(json!([
            {"model_name": "g", "litellm_params": {"model": "openai/a"}, "model_info": {
                "id": "a", "enforce_supported_endpoints": true, "supported_endpoints": ["/v1/embeddings"],
            }},
        ]));
        let caller = Caller {
            endpoint: Some("/v1/chat/completions"),
            requested: "a",
            ..Caller::default()
        };
        let refused = resolve_request(&registry, "a", false, caller, None, None, Timestamp::now());
        assert!(matches!(
            refused,
            Err(Error::EndpointNotSupported { model, .. }) if model == "openai/a"
        ));
    }
}
