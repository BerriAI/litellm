//! Whether every deployment a request can reach lets a caller set an override. The router picks
//! the deployment that serves a request, and a failure moves it on to a fallback, so an opt-in
//! counts only when every deployment the request can land on carries it.
//!
//! Like `check.rs`, each function keeps the name its Python original had.

use fancy_regex::Regex;

use super::{names::VERTEX_TARGET_PARAMS, value::CallerValue};

pub(crate) const MAX_OPT_IN_FALLBACK_MODELS: usize = 100;

/// Providers that read a Vertex project and location from the call.
const VERTEX_TARGET_READING_PROVIDERS: [&str; 4] = ["vertex_ai", "vertex_ai_beta", "gemini", "gdc"];

/// One deployment of a model group a request can reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deployment {
    pub model_name: String,
    /// `None` when the deployment's provider cannot be told.
    pub provider: Option<String>,
    /// `configurable_clientside_auth_params`, or `None` when the deployment sets none.
    pub opt_in: Option<OptIn>,
}

/// The overrides a deployment lets a caller set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OptIn {
    pub params: Vec<String>,
    /// The `api_base` entries, each matched as a Python `re.match` pattern or as written.
    pub api_base_patterns: Vec<String>,
}

/// The model groups a request can reach, each as the deployments that serve it. The caller
/// resolves them from [`super::Needs::models`], as Python's `_opt_in_model_groups` does: each name
/// through its alias, the router's fallbacks of every kind from each, then the groups the
/// deployments found route under, until no new group turns up. A name no deployment serves is an
/// empty group, and a proxy that routes no deployment hands in none, which grants nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reachable {
    pub groups: Vec<Vec<Deployment>>,
}

impl Reachable {
    /// Whether every group lets a caller set `param` to `value`. A Vertex project or location
    /// only has to be allowed by the groups whose provider reads one.
    pub(crate) fn allow(&self, param: &str, value: &CallerValue) -> bool {
        if self.groups.is_empty() {
            return false;
        }
        let vertex_target_only = VERTEX_TARGET_PARAMS.contains(&param);
        self.groups
            .iter()
            .filter(|group| !vertex_target_only || model_group_reads_vertex_target(group))
            .all(|group| allow_model_level_clientside_configurable_parameters(group, param, value))
    }
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
/// only when every deployment of the model group carries it. A group with no deployment allows
/// nothing.
fn allow_model_level_clientside_configurable_parameters(
    deployments: &[Deployment],
    param: &str,
    request_body_value: &CallerValue,
) -> bool {
    !deployments.is_empty()
        && deployments.iter().all(|deployment| {
            is_param_allowed(param, request_body_value, deployment.opt_in.as_ref())
        })
}

fn is_param_allowed(
    param: &str,
    request_body_value: &CallerValue,
    configurable_clientside_auth_params: Option<&OptIn>,
) -> bool {
    let Some(opt_in) = configurable_clientside_auth_params else {
        return false;
    };
    if opt_in.params.iter().any(|item| item == param) {
        return true;
    }
    let (true, Some(api_base)) = (param == "api_base", request_body_value.text()) else {
        return false;
    };
    opt_in
        .api_base_patterns
        .iter()
        .any(|pattern| check_regex_or_str_match(api_base, pattern))
}

/// `check_regex_or_str_match`: Python's `re.match`, anchored at the start only, or the pattern as
/// written. A pattern the Rust dialect cannot compile, or a match that runs out of backtracking,
/// matches nothing but itself.
fn check_regex_or_str_match(request_body_value: &str, regex_str: &str) -> bool {
    regex_str == request_body_value
        || Regex::new(&format!(r"\A(?:{regex_str})"))
            .ok()
            .and_then(|pattern| pattern.is_match(request_body_value).ok())
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn deployment(provider: &str, params: &[&str]) -> Deployment {
        Deployment {
            model_name: "group".to_owned(),
            provider: Some(provider.to_owned()),
            opt_in: Some(OptIn {
                params: params.iter().map(|param| (*param).to_owned()).collect(),
                api_base_patterns: vec![r"https://allowed\.example/.*".to_owned()],
            }),
        }
    }

    fn grants(groups: Vec<Vec<Deployment>>, param: &str, value: &str) -> bool {
        Reachable { groups }.allow(param, &CallerValue::Text(value.to_owned()))
    }

    #[rstest]
    #[case::every_deployment_opts_in(&["client_id"], &["client_id"], true)]
    #[case::one_deployment_does_not(&["client_id"], &[], false)]
    fn a_group_grants_only_what_every_deployment_opts_in_to(
        #[case] first: &[&str],
        #[case] second: &[&str],
        #[case] expected: bool,
    ) {
        let group = vec![deployment("azure", first), deployment("azure", second)];

        assert_eq!(grants(vec![group], "client_id", "caller"), expected);
    }

    #[rstest]
    #[case::fallback_opts_in(&["client_id"], true)]
    #[case::fallback_does_not(&[], false)]
    fn every_reachable_group_must_opt_in(#[case] fallback_params: &[&str], #[case] expected: bool) {
        let groups = vec![
            vec![deployment("azure", &["client_id"])],
            vec![deployment("azure", fallback_params)],
        ];

        assert_eq!(grants(groups, "client_id", "caller"), expected);
    }

    #[rstest]
    #[case::no_group(vec![])]
    #[case::a_group_no_deployment_serves(vec![vec![]])]
    #[case::a_deployment_without_an_opt_in(vec![vec![Deployment { opt_in: None, ..deployment("azure", &[]) }]])]
    fn nothing_is_granted_without_a_deployment_that_opts_in(#[case] groups: Vec<Vec<Deployment>>) {
        assert!(!grants(groups, "client_id", "caller"));
    }

    #[rstest]
    #[case::matched_from_the_start("https://allowed.example/v1", true)]
    #[case::only_the_start_is_anchored(
        "https://allowed.example/v1?x=https://attacker.example",
        true
    )]
    #[case::matched_later_in_the_value(
        "https://attacker.example/https://allowed.example/v1",
        false
    )]
    #[case::other_host("https://attacker.example/v1", false)]
    fn an_api_base_is_granted_when_a_pattern_matches_it_as_re_match_does(
        #[case] value: &str,
        #[case] expected: bool,
    ) {
        let groups = vec![vec![deployment("openai", &[])]];

        assert_eq!(grants(groups, "api_base", value), expected);
    }

    #[rstest]
    #[case::written_out("https://own.example/(", "https://own.example/(", true)]
    #[case::uncompilable("https://own.example/(", "https://own.example/x", false)]
    #[case::python_lookahead(
        r"https://(?!internal)[a-z]+\.example",
        "https://public.example",
        true
    )]
    fn a_pattern_is_read_in_pythons_regex_dialect_or_as_written(
        #[case] pattern: &str,
        #[case] value: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(check_regex_or_str_match(value, pattern), expected);
    }

    #[rstest]
    #[case::other_provider_is_not_asked("openai", true)]
    #[case::vertex_must_opt_in("vertex_ai", false)]
    fn a_vertex_target_is_judged_only_by_groups_that_read_one(
        #[case] fallback_provider: &str,
        #[case] expected: bool,
    ) {
        let groups = vec![
            vec![deployment("vertex_ai", &["vertex_location"])],
            vec![deployment(fallback_provider, &[])],
        ];

        assert_eq!(grants(groups, "vertex_location", "us-east5"), expected);
    }
}
