//! `enable_pre_call_checks` (`_pre_call_checks`): before a pick, drop the deployments whose
//! context window the prompt exceeds, those outside the request's `allowed_model_region`, and
//! those that cannot take a requested param, then decide which error a group with nothing left
//! raises. Each stage is its own module. The host reports what it alone can read, as it reads it:
//! [`RouterHost::pre_call_facts`] the prompt's token count, each deployment's context window, the
//! live `litellm.drop_params` and how the request sends the checked params, and
//! [`RouterHost::unsupported_params`] which deployments do not support the params [`requested`]
//! asks for. Every decision is made here.
//!
//! [`RouterHost::pre_call_facts`]: crate::host::RouterHost::pre_call_facts
//! [`RouterHost::unsupported_params`]: crate::host::RouterHost::unsupported_params

mod context_window;
mod params;
mod region;

use std::{collections::BTreeMap, sync::Arc};

use context_window::Overflow;
use litellm_python_compat::lenient::LaxBool;
use litellm_router_types::CallSettings;
use serde::Deserialize;
use serde_json::Value;
use serde_with::serde_as;

pub use params::{CHECKED_PARAMS, Sent, Unsupported, requested};

use crate::{error::Error, registry::Deployment};

/// What the host knows about one deployment for the pre-call checks.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct DeploymentFacts {
    /// The deployment's context window (`get_router_model_info`'s `max_input_tokens`), when it
    /// declares one and resolves.
    pub max_input_tokens: Option<i64>,
}

/// What the host knows about a request for the pre-call checks.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct PreCallFacts {
    /// The request carries `messages` or `input`: without either no check runs.
    pub has_input: bool,
    /// The prompt may be counted against a context window (no compaction is pending).
    pub countable: bool,
    /// The prompt's token count, when it was counted.
    pub input_tokens: Option<i64>,
    /// The request's `allowed_model_region`.
    pub allowed_model_region: Option<Value>,
    /// The live `litellm.drop_params`. Params are checked only while it is `false`.
    #[serde_as(as = "LaxBool")]
    pub drop_params: Option<bool>,
    /// How the request sends each of [`CHECKED_PARAMS`] it carries.
    pub sent_params: BTreeMap<String, Sent>,
    /// The request's own call settings.
    pub call_settings: CallSettings,
    /// Facts by deployment id, for the candidates and the group's deployments.
    pub deployments: BTreeMap<String, DeploymentFacts>,
}

impl PreCallFacts {
    fn of(&self, deployment: &Deployment) -> DeploymentFacts {
        self.deployments
            .get(&deployment.id)
            .copied()
            .unwrap_or_default()
    }
}

/// `_pre_call_checks` over `candidates`; `pool` is the group's deployments before health filters
/// (`request_deployments`), which decide whether nothing left is an unsupported-params error.
/// `unsupported` is the host's answer for the params [`requested`] asks of `facts`.
pub fn filter(
    model: &str,
    candidates: Vec<Arc<Deployment>>,
    pool: &[Arc<Deployment>],
    facts: &PreCallFacts,
    unsupported: &Unsupported,
) -> Result<Vec<Arc<Deployment>>, Error> {
    if !facts.has_input {
        return Ok(candidates);
    }
    let overflows: Vec<Option<Overflow>> = candidates
        .iter()
        .map(|deployment| context_window::overflow(deployment, facts))
        .collect();
    let context_window_error = overflows.iter().any(Option::is_some);
    let dropped: Vec<bool> = candidates
        .iter()
        .zip(&overflows)
        .map(|(deployment, overflow)| overflow.is_some() || region::outside(deployment, facts))
        .collect();
    let params = params::Check::of(facts, unsupported);
    let left: Vec<&Deployment> = candidates
        .iter()
        .zip(&dropped)
        .filter(|(_, dropped)| !**dropped)
        .map(|(deployment, _)| &**deployment)
        .collect();
    let mut failing = params.failing(&left).into_iter();
    let kept: Vec<bool> = dropped
        .iter()
        .map(|dropped| !dropped && failing.next() == Some(false))
        .collect();
    if !kept.contains(&true) {
        if !context_window_error && let Some(error) = params.unsupported(model, pool) {
            return Err(error);
        }
        if context_window_error {
            return Err(context_window::exceeded(model, overflows.iter().flatten()));
        }
    }
    Ok(candidates
        .into_iter()
        .zip(kept)
        .filter_map(|(deployment, kept)| kept.then_some(deployment))
        .collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::registry::DeploymentSpec;

    fn deployment(id: &str, params: Value, model_info: Value) -> Arc<Deployment> {
        let mut info = model_info;
        info["id"] = json!(id);
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "g", "litellm_params": params, "model_info": info,
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    fn facts(entries: &[(&str, DeploymentFacts)]) -> PreCallFacts {
        PreCallFacts {
            has_input: true,
            countable: true,
            deployments: entries
                .iter()
                .map(|(id, f)| ((*id).to_owned(), *f))
                .collect(),
            ..PreCallFacts::default()
        }
    }

    fn window(limit: i64) -> DeploymentFacts {
        DeploymentFacts {
            max_input_tokens: Some(limit),
        }
    }

    #[rstest]
    fn a_prompt_over_a_context_window_drops_that_deployment() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let big = deployment("big", json!({"model": "openai/b"}), json!({}));
        let mut found = facts(&[("small", window(5)), ("big", window(10_000))]);
        found.input_tokens = Some(100);
        let kept = filter("g", vec![small, big], &[], &found, &Unsupported::default()).unwrap();
        assert_eq!(
            kept.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["big"]
        );
    }

    #[rstest]
    fn no_window_left_names_each_deployment_back_to_back() {
        let one = deployment("one", json!({"model": "openai/one"}), json!({}));
        let two = deployment(
            "two",
            json!({"model": "openai/two"}),
            json!({"base_model": "azure/base"}),
        );
        let mut found = facts(&[("one", window(5)), ("two", window(6))]);
        found.input_tokens = Some(9);
        let error = filter("g", vec![one, two], &[], &found, &Unsupported::default()).unwrap_err();
        let Error::ContextWindowExceeded { model, detail } = error else {
            panic!("expected a context window error, got {error:?}");
        };
        assert_eq!(model, "g");
        assert_eq!(
            detail,
            "Model=openai/one, Max Input Tokens=5, Got=9Model=azure/base, Max Input Tokens=6, Got=9"
        );
    }

    #[rstest]
    fn without_input_nothing_is_checked() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[("small", window(5))]);
        found.input_tokens = Some(100);
        found.has_input = false;
        assert_eq!(
            filter("g", vec![small], &[], &found, &Unsupported::default())
                .unwrap()
                .len(),
            1
        );
    }

    #[rstest]
    fn a_prompt_that_may_not_be_counted_fits_every_window() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[("small", window(5))]);
        found.input_tokens = Some(100);
        found.countable = false;
        assert_eq!(
            filter("g", vec![small], &[], &found, &Unsupported::default())
                .unwrap()
                .len(),
            1
        );
    }

    #[rstest]
    #[case::in_region(json!("eu"), &["eu"])]
    #[case::out_of_region(json!("ap"), &[])]
    #[case::null_region(Value::Null, &["eu", "none"])]
    fn the_allowed_region_keeps_only_its_deployments(#[case] region: Value, #[case] kept: &[&str]) {
        let eu = deployment(
            "eu",
            json!({"model": "openai/a", "region_name": "eu"}),
            json!({}),
        );
        let none = deployment("none", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[]);
        found.allowed_model_region = Some(region);
        let left = filter("g", vec![eu, none], &[], &found, &Unsupported::default()).unwrap();
        assert_eq!(left.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(), kept);
    }

    const NO_FACTS: &[(&str, DeploymentFacts)] = &[];

    fn sending(response_format: Sent) -> PreCallFacts {
        PreCallFacts {
            drop_params: Some(false),
            sent_params: BTreeMap::from([("response_format".to_owned(), response_format)]),
            ..facts(NO_FACTS)
        }
    }

    fn lacking(ids: &[&str]) -> Unsupported {
        ids.iter()
            .map(|id| ((*id).to_owned(), ["response_format".to_owned()].into()))
            .collect()
    }

    fn names(names: &[&str]) -> Option<Box<[String]>> {
        Some(names.iter().map(|name| (*name).to_owned()).collect())
    }

    fn ids(kept: &[Arc<Deployment>]) -> Vec<&str> {
        kept.iter().map(|d| d.id.as_str()).collect()
    }

    #[rstest]
    #[case::structured(Sent::Structured, Some(false), true, &["response_format"])]
    #[case::text(Sent::Text, Some(false), true, &[])]
    #[case::null(Sent::Null, Some(false), true, &[])]
    #[case::global_drop_params(Sent::Structured, Some(true), true, &[])]
    #[case::global_drop_params_unset(Sent::Structured, None, true, &[])]
    #[case::no_input(Sent::Structured, Some(false), false, &[])]
    fn a_param_is_requested_only_as_a_structured_value_with_drop_params_off(
        #[case] response_format: Sent,
        #[case] global_drop_params: Option<bool>,
        #[case] has_input: bool,
        #[case] asked: &[&str],
    ) {
        let found = PreCallFacts {
            drop_params: global_drop_params,
            has_input,
            ..sending(response_format)
        };
        assert_eq!(requested(&found), asked);
    }

    #[rstest]
    fn a_param_outside_the_checked_ones_is_never_requested() {
        let found = PreCallFacts {
            sent_params: BTreeMap::from([("tools".to_owned(), Sent::Structured)]),
            ..sending(Sent::Null)
        };
        assert!(requested(&found).is_empty());
    }

    #[rstest]
    fn a_param_that_is_not_requested_drops_no_deployment_that_lacks_it() {
        let old = deployment("old", json!({"model": "openai/old"}), json!({}));
        let new = deployment("new", json!({"model": "openai/new"}), json!({}));
        let found = sending(Sent::Text);
        let kept = filter("g", vec![old, new], &[], &found, &lacking(&["old"])).unwrap();
        assert_eq!(ids(&kept), ["old", "new"]);
    }

    #[rstest]
    fn params_are_checked_only_on_deployments_inside_the_region() {
        let eu = deployment(
            "eu",
            json!({"model": "openai/old", "region_name": "eu", "drop_params": true}),
            json!({}),
        );
        let ap = deployment(
            "ap",
            json!({"model": "openai/new", "region_name": "ap"}),
            json!({}),
        );
        let mut found = sending(Sent::Structured);
        found.allowed_model_region = Some(json!("eu"));
        let kept = filter("g", vec![eu, ap], &[], &found, &lacking(&["eu"])).unwrap();
        assert_eq!(ids(&kept), ["eu"]);
    }

    #[rstest]
    fn a_group_where_every_deployment_rejects_the_param_is_unsupported() {
        let old = deployment("old", json!({"model": "openai/old"}), json!({}));
        let found = sending(Sent::Structured);
        let error = filter("g", vec![old.clone()], &[old], &found, &lacking(&["old"])).unwrap_err();
        assert!(
            matches!(error, Error::UnsupportedParams { params, .. } if params == ["response_format"])
        );
    }

    #[rstest]
    fn a_deployment_that_takes_the_param_wins_over_one_that_would_drop_it() {
        let old = deployment(
            "old",
            json!({"model": "openai/old", "drop_params": true}),
            json!({}),
        );
        let new = deployment("new", json!({"model": "openai/new"}), json!({}));
        let found = sending(Sent::Structured);
        let kept = filter("g", vec![old, new], &[], &found, &lacking(&["old"])).unwrap();
        assert_eq!(ids(&kept), ["new"]);
    }

    fn drops() -> CallSettings {
        CallSettings {
            drop_params: Some(true),
            ..CallSettings::default()
        }
    }

    fn allows(params: &[&str]) -> CallSettings {
        CallSettings {
            allowed_openai_params: names(params),
            ..CallSettings::default()
        }
    }

    fn drops_only(params: &[&str]) -> CallSettings {
        CallSettings {
            additional_drop_params: names(params),
            ..CallSettings::default()
        }
    }

    #[rstest]
    #[case::rejected(json!({}), CallSettings::default(), 0)]
    #[case::own_drop_params(json!({"drop_params": true}), CallSettings::default(), 1)]
    #[case::own_allowlist(json!({"allowed_openai_params": ["response_format"]}), CallSettings::default(), 1)]
    #[case::own_drop_list(json!({"additional_drop_params": ["response_format"]}), CallSettings::default(), 1)]
    #[case::sent_drop_params(json!({}), drops(), 1)]
    #[case::sent_allowlist(json!({}), allows(&["tools", "response_format"]), 1)]
    #[case::sent_drop_list(json!({}), drops_only(&["response_format"]), 1)]
    #[case::sent_drop_params_off_replaces_own(
        json!({"drop_params": true}),
        CallSettings { drop_params: Some(false), ..CallSettings::default() },
        0
    )]
    #[case::sent_allowlist_replaces_own(
        json!({"allowed_openai_params": ["response_format"]}),
        allows(&["tools"]),
        0
    )]
    #[case::sent_empty_allowlist_replaces_own(
        json!({"allowed_openai_params": ["response_format"]}),
        allows(&[]),
        0
    )]
    #[case::sent_empty_drop_list_replaces_own(
        json!({"additional_drop_params": ["response_format"]}),
        drops_only(&[]),
        0
    )]
    #[case::sent_other_key_keeps_own(json!({"drop_params": true}), allows(&["tools"]), 1)]
    fn when_every_deployment_lacks_the_param_only_those_rejecting_it_are_dropped(
        #[case] own: Value,
        #[case] sent: CallSettings,
        #[case] kept: usize,
    ) {
        let mut params = own;
        params["model"] = json!("openai/old");
        let old = deployment("old", params, json!({}));
        let found = PreCallFacts {
            call_settings: sent,
            ..sending(Sent::Structured)
        };
        let left = filter("g", vec![old], &[], &found, &lacking(&["old"])).unwrap();
        assert_eq!(left.len(), kept);
    }
}
