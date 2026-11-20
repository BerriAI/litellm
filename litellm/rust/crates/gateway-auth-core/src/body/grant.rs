//! Which deployments a request can reach, and whether all of them let a caller set an override.
//! The router picks the deployment that serves a request, and a failure moves it on to a
//! fallback, so an opt-in counts only when every deployment the request can land on carries it.
//!
//! Like `check.rs`, each function keeps the name its Python original had.

use std::collections::BTreeSet;

use super::{
    check::Failure,
    host::{Deployment, Deployments, OptIn},
    names::VERTEX_TARGET_PARAMS,
    value::CallerValue,
};
use crate::error::body::Rejection;

pub(crate) const MAX_OPT_IN_FALLBACK_MODELS: usize = 100;

/// Providers that read a Vertex project and location from the call.
const VERTEX_TARGET_READING_PROVIDERS: [&str; 4] = ["vertex_ai", "vertex_ai_beta", "gemini", "gdc"];

/// The model groups a request can reach, each as the deployments that serve it. Python kept the
/// group names and read their deployments again for every override.
#[derive(Debug)]
pub(crate) struct OptInGroups {
    every: Vec<Vec<Deployment>>,
}

impl OptInGroups {
    /// Whether every group lets a caller set `param` to `value`. A Vertex project or location
    /// only has to be allowed by the groups whose provider reads one.
    pub(crate) fn allow<D: Deployments>(
        &self,
        router: Option<&D>,
        param: &str,
        value: &CallerValue,
    ) -> Result<bool, D::Error> {
        let vertex_target_only = VERTEX_TARGET_PARAMS.contains(&param);
        for model_group in &self.every {
            if vertex_target_only && !model_group_reads_vertex_target(model_group) {
                continue;
            }
            if !allow_model_level_clientside_configurable_parameters(
                router,
                model_group,
                param,
                value,
            )? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The groups `model` and `also_requested_models` can reach. A proxy that routes no deployment
/// reaches each requested name, with no deployment to allow anything.
pub(crate) fn resolve_opt_in_groups<D: Deployments>(
    model: &str,
    also_requested_models: &BTreeSet<&str>,
    router: Option<&D>,
) -> Result<OptInGroups, Failure<D::Error>> {
    if also_requested_models.len() > MAX_OPT_IN_FALLBACK_MODELS {
        return Err(Rejection::TooManyFallbackModels {
            limit: MAX_OPT_IN_FALLBACK_MODELS,
        }
        .into());
    }
    let requested: BTreeSet<String> = also_requested_models
        .iter()
        .copied()
        .chain([model])
        .map(str::to_owned)
        .collect();
    let every = match router {
        Some(router) => opt_in_model_groups(router, &requested).map_err(Failure::Host)?,
        None => requested.iter().map(|_| Vec::new()).collect(),
    };
    Ok(OptInGroups { every })
}

/// Follows aliases, the router's fallbacks, and the groups each found deployment routes under,
/// until no new group turns up.
fn opt_in_model_groups<D: Deployments>(
    router: &D,
    requested: &BTreeSet<String>,
) -> Result<Vec<Vec<Deployment>>, D::Error> {
    let routed = requested
        .iter()
        .map(|name| resolve_alias(router, name))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut groups = routed.clone();
    for name in requested.union(&routed) {
        groups.extend(fallback_groups(router, name)?);
    }

    let mut every = Vec::new();
    let mut reached = groups.clone();
    let mut frontier = groups;
    while !frontier.is_empty() {
        frontier = routed_opt_in_hop(router, &frontier, &mut reached, &mut every)?;
    }
    Ok(every)
}

/// One hop of the walk: reads the deployments of each group in `frontier` into `every`, and
/// returns the groups they and their fallbacks route under that were not reached before.
fn routed_opt_in_hop<D: Deployments>(
    router: &D,
    frontier: &BTreeSet<String>,
    reached: &mut BTreeSet<String>,
    every: &mut Vec<Vec<Deployment>>,
) -> Result<BTreeSet<String>, D::Error> {
    let mut routed_groups = BTreeSet::new();
    for model in frontier {
        let deployments = router.group(model)?;
        routed_groups.extend(
            deployments
                .iter()
                .map(|deployment| deployment.model_name.clone())
                .filter(|name| !reached.contains(name)),
        );
        every.push(deployments);
    }
    let mut next_hop = routed_groups.clone();
    for name in &routed_groups {
        next_hop.extend(fallback_groups(router, name)?);
    }
    next_hop.retain(|name| !reached.contains(name));
    reached.extend(next_hop.iter().cloned());
    Ok(next_hop)
}

fn resolve_alias<D: Deployments>(router: &D, name: &str) -> Result<String, D::Error> {
    Ok(router.alias(name)?.unwrap_or_else(|| name.to_owned()))
}

fn fallback_groups<D: Deployments>(router: &D, name: &str) -> Result<Vec<String>, D::Error> {
    router
        .fallbacks(name)?
        .iter()
        .map(|fallback| resolve_alias(router, fallback))
        .collect()
}

fn model_group_reads_vertex_target(deployments: &[Deployment]) -> bool {
    deployments.is_empty()
        || deployments.iter().any(|deployment| {
            deployment
                .provider
                .as_deref()
                .is_none_or(|provider| VERTEX_TARGET_READING_PROVIDERS.contains(&provider))
        })
}

/// The router, not the caller, picks which deployment serves the request, so an opt-in counts
/// only when every deployment of the model group carries it. A group with no deployment, or a
/// proxy with no router, allows nothing.
fn allow_model_level_clientside_configurable_parameters<D: Deployments>(
    router: Option<&D>,
    deployments: &[Deployment],
    param: &str,
    request_body_value: &CallerValue,
) -> Result<bool, D::Error> {
    let Some(router) = router else {
        return Ok(false);
    };
    if deployments.is_empty() {
        return Ok(false);
    }
    for deployment in deployments {
        if !is_param_allowed(
            router,
            param,
            request_body_value,
            deployment.opt_in.as_ref(),
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn is_param_allowed<D: Deployments>(
    router: &D,
    param: &str,
    request_body_value: &CallerValue,
    configurable_clientside_auth_params: Option<&OptIn>,
) -> Result<bool, D::Error> {
    let Some(opt_in) = configurable_clientside_auth_params else {
        return Ok(false);
    };
    if opt_in.params.iter().any(|item| item == param) {
        return Ok(true);
    }
    let (true, Some(api_base)) = (param == "api_base", request_body_value.text()) else {
        return Ok(false);
    };
    for pattern in &opt_in.api_base_patterns {
        if router.api_base_matches(pattern, api_base)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, convert::Infallible};

    use rstest::rstest;

    use super::*;

    #[derive(Default)]
    struct Router {
        groups: BTreeMap<&'static str, Vec<Deployment>>,
        aliases: BTreeMap<&'static str, &'static str>,
        fallbacks: BTreeMap<&'static str, Vec<&'static str>>,
    }

    impl Deployments for Router {
        type Error = Infallible;

        fn group(&self, model: &str) -> Result<Vec<Deployment>, Infallible> {
            Ok(self.groups.get(model).cloned().unwrap_or_default())
        }

        fn alias(&self, name: &str) -> Result<Option<String>, Infallible> {
            Ok(self.aliases.get(name).map(|group| (*group).to_owned()))
        }

        fn fallbacks(&self, name: &str) -> Result<Vec<String>, Infallible> {
            Ok(self
                .fallbacks
                .get(name)
                .map(|names| names.iter().map(|name| (*name).to_owned()).collect())
                .unwrap_or_default())
        }

        fn api_base_matches(&self, pattern: &str, value: &str) -> Result<bool, Infallible> {
            Ok(pattern == value)
        }
    }

    fn deployment(model_name: &str, provider: &str, params: &[&str]) -> Deployment {
        Deployment {
            model_name: model_name.to_owned(),
            provider: Some(provider.to_owned()),
            opt_in: Some(OptIn {
                params: params.iter().map(|param| (*param).to_owned()).collect(),
                api_base_patterns: vec!["https://allowed.example".to_owned()],
            }),
        }
    }

    fn grants(router: &Router, model: &str, param: &str, value: &str) -> bool {
        let requested = BTreeSet::from([model.to_owned()]);
        let Ok(every) = opt_in_model_groups(router, &requested);
        let Ok(granted) =
            OptInGroups { every }.allow(Some(router), param, &CallerValue::Text(value.to_owned()));
        granted
    }

    #[rstest]
    #[case::every_deployment_opts_in(&["client_id"], &["client_id"], true)]
    #[case::one_deployment_does_not(&["client_id"], &[], false)]
    fn a_group_grants_only_what_every_deployment_opts_in_to(
        #[case] first: &[&str],
        #[case] second: &[&str],
        #[case] expected: bool,
    ) {
        let router = Router {
            groups: BTreeMap::from([(
                "azure-group",
                vec![
                    deployment("azure-group", "azure", first),
                    deployment("azure-group", "azure", second),
                ],
            )]),
            ..Router::default()
        };

        assert_eq!(
            grants(&router, "azure-group", "client_id", "caller"),
            expected
        );
    }

    #[rstest]
    #[case::fallback_opts_in(&["client_id"], true)]
    #[case::fallback_does_not(&[], false)]
    fn a_fallback_group_must_opt_in_too(#[case] fallback_params: &[&str], #[case] expected: bool) {
        let router = Router {
            groups: BTreeMap::from([
                (
                    "primary",
                    vec![deployment("primary", "azure", &["client_id"])],
                ),
                (
                    "backup",
                    vec![deployment("backup", "azure", fallback_params)],
                ),
            ]),
            fallbacks: BTreeMap::from([("primary", vec!["backup"])]),
            ..Router::default()
        };

        assert_eq!(grants(&router, "primary", "client_id", "caller"), expected);
    }

    #[rstest]
    fn an_alias_is_judged_by_the_group_it_names() {
        let router = Router {
            groups: BTreeMap::from([(
                "primary",
                vec![deployment("primary", "azure", &["client_id"])],
            )]),
            aliases: BTreeMap::from([("nickname", "primary")]),
            ..Router::default()
        };

        assert!(grants(&router, "nickname", "client_id", "caller"));
    }

    #[rstest]
    fn a_group_reached_only_through_a_deployment_must_opt_in_too() {
        let router = Router {
            groups: BTreeMap::from([
                (
                    "openai/*",
                    vec![deployment("wildcard-group", "openai", &["client_id"])],
                ),
                (
                    "wildcard-group",
                    vec![deployment("wildcard-group", "openai", &[])],
                ),
            ]),
            ..Router::default()
        };

        assert!(!grants(&router, "openai/*", "client_id", "caller"));
    }

    #[rstest]
    fn a_model_no_deployment_serves_grants_nothing() {
        assert!(!grants(
            &Router::default(),
            "unknown",
            "client_id",
            "caller"
        ));
    }

    #[rstest]
    #[case::listed_pattern("https://allowed.example", true)]
    #[case::other_host("https://attacker.example", false)]
    fn an_api_base_is_granted_when_a_pattern_admits_it(
        #[case] value: &str,
        #[case] expected: bool,
    ) {
        let router = Router {
            groups: BTreeMap::from([("primary", vec![deployment("primary", "openai", &[])])]),
            ..Router::default()
        };

        assert_eq!(grants(&router, "primary", "api_base", value), expected);
    }

    #[rstest]
    #[case::other_provider_is_not_asked("openai", true)]
    #[case::vertex_must_opt_in("vertex_ai", false)]
    fn a_vertex_target_is_judged_only_by_groups_that_read_one(
        #[case] fallback_provider: &str,
        #[case] expected: bool,
    ) {
        let router = Router {
            groups: BTreeMap::from([
                (
                    "primary",
                    vec![deployment("primary", "vertex_ai", &["vertex_location"])],
                ),
                ("backup", vec![deployment("backup", fallback_provider, &[])]),
            ]),
            fallbacks: BTreeMap::from([("primary", vec!["backup"])]),
            ..Router::default()
        };

        assert_eq!(
            grants(&router, "primary", "vertex_location", "us-east5"),
            expected
        );
    }
}
