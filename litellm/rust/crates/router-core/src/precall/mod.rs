//! `enable_pre_call_checks` (`_pre_call_checks`): before a pick, drop the deployments whose
//! context window the prompt exceeds, those outside the request's `allowed_model_region`, and
//! those that cannot take a requested param, then decide which error a group with nothing left
//! raises. Each stage is its own module. The facts that need live model data (the prompt's token
//! count, each deployment's context window, which params it takes) come from
//! [`RouterHost::pre_call_facts`]; every decision is made here.
//!
//! [`RouterHost::pre_call_facts`]: crate::host::RouterHost::pre_call_facts

mod context_window;
mod params;
mod region;

use std::{collections::BTreeMap, sync::Arc};

use context_window::Overflow;
pub(crate) use context_window::named_model;
use serde::Deserialize;
use serde_json::Value;

use crate::{error::Error, registry::Deployment};

/// What the host knows about one deployment for the pre-call checks.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct DeploymentFacts {
    /// The deployment's context window (`get_router_model_info`'s `max_input_tokens`), when it
    /// declares one and resolves.
    pub max_input_tokens: Option<i64>,
    /// The prompt's token count with this deployment's tokenizer, an inline PDF priced the way it
    /// bills one, when the host counted it.
    pub input_tokens: Option<i64>,
    /// The count came from a stand-in tokenizer, not the one the provider bills with.
    pub estimated: bool,
    /// `_deployment_lacks_params`: it does not support one of the requested params.
    pub lacks_params: bool,
    /// `_deployment_rejects_params`: it lacks one the request and deployment neither drop nor allow.
    pub rejects_params: bool,
}

/// What the host knows about a request for the pre-call checks.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct PreCallFacts {
    /// The request carries `messages` or `input`: without either no check runs.
    pub has_input: bool,
    /// The prompt may be counted against a context window (no compaction is pending).
    pub countable: bool,
    /// The request's `allowed_model_region`.
    pub allowed_model_region: Option<Value>,
    /// The requested params the check filters on, sorted (`_pre_call_check_requested_chat_params`).
    pub requested_params: Vec<String>,
    /// Facts by deployment id, for the candidates and the group's deployments.
    pub deployments: BTreeMap<String, DeploymentFacts>,
}

impl PreCallFacts {
    pub(crate) fn max_input_tokens(&self, deployment_id: &str) -> Option<i64> {
        self.deployments
            .get(deployment_id)
            .and_then(|facts| facts.max_input_tokens)
    }

    fn of(&self, deployment: &Deployment) -> DeploymentFacts {
        self.deployments
            .get(&deployment.id)
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Debug, Default)]
pub struct Filtered {
    pub kept: Vec<Arc<Deployment>>,
    pub uncertain: BTreeMap<String, String>,
}

/// `_pre_call_checks` over `candidates`; `pool` is the group's deployments before health filters
/// (`request_deployments`), which decide whether nothing left is an unsupported-params error.
pub fn filter(
    model: &str,
    candidates: Vec<Arc<Deployment>>,
    pool: &[Arc<Deployment>],
    facts: &PreCallFacts,
) -> Result<Filtered, Error> {
    if !facts.has_input {
        return Ok(Filtered {
            kept: candidates,
            ..Filtered::default()
        });
    }
    let overflows: Vec<Option<Overflow>> = candidates
        .iter()
        .map(|deployment| context_window::overflow(deployment, facts))
        .collect();
    let certain = |overflow: &Option<Overflow>| overflow.as_ref().is_some_and(Overflow::certain);
    let context_window_error = overflows.iter().any(certain);
    let dropped: Vec<bool> = candidates
        .iter()
        .zip(&overflows)
        .map(|(deployment, overflow)| certain(overflow) || region::outside(deployment, facts))
        .collect();
    let left: Vec<&Deployment> = candidates
        .iter()
        .zip(&dropped)
        .filter(|(_, dropped)| !**dropped)
        .map(|(deployment, _)| &**deployment)
        .collect();
    let mut failing = params::failing(&left, facts).into_iter();
    let kept: Vec<bool> = dropped
        .iter()
        .map(|dropped| !dropped && failing.next() == Some(false))
        .collect();
    if !kept.contains(&true) {
        if !context_window_error && let Some(error) = params::unsupported(model, pool, facts) {
            return Err(error);
        }
        if context_window_error {
            return Err(context_window::exceeded(
                model,
                overflows
                    .iter()
                    .flatten()
                    .filter(|overflow| overflow.certain()),
            ));
        }
    }
    let uncertain = candidates
        .iter()
        .zip(&overflows)
        .filter_map(|(deployment, overflow)| {
            overflow
                .as_ref()
                .filter(|overflow| !overflow.certain())
                .map(|overflow| (deployment.id.clone(), overflow.detail()))
        })
        .collect();
    Ok(Filtered {
        kept: candidates
            .into_iter()
            .zip(kept)
            .filter_map(|(deployment, kept)| kept.then_some(deployment))
            .collect(),
        uncertain,
    })
}

pub fn settle(
    model: &str,
    healthy: Vec<Arc<Deployment>>,
    uncertain: &BTreeMap<String, String>,
    refuse_uncertain: bool,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let (unsure, fitting): (Vec<_>, Vec<_>) = healthy
        .into_iter()
        .partition(|deployment| uncertain.contains_key(&deployment.id));
    if !fitting.is_empty() || unsure.is_empty() {
        return Ok(fitting);
    }
    if refuse_uncertain {
        return Err(Error::ContextWindowExceeded {
            model: model.to_owned(),
            detail: unsure
                .iter()
                .filter_map(|deployment| uncertain.get(&deployment.id).map(String::as_str))
                .collect(),
            held_for_fallback: true,
        });
    }
    Ok(unsure)
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
                .map(|(id, f)| ((*id).to_owned(), f.clone()))
                .collect(),
            ..PreCallFacts::default()
        }
    }

    fn counted(limit: i64, tokens: i64, estimated: bool) -> DeploymentFacts {
        DeploymentFacts {
            max_input_tokens: Some(limit),
            input_tokens: Some(tokens),
            estimated,
            ..DeploymentFacts::default()
        }
    }

    fn ids(deployments: &[Arc<Deployment>]) -> Vec<&str> {
        deployments.iter().map(|d| d.id.as_str()).collect()
    }

    #[rstest]
    fn a_prompt_over_a_context_window_drops_that_deployment() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let big = deployment("big", json!({"model": "openai/b"}), json!({}));
        let found = facts(&[
            ("small", counted(5, 100, false)),
            ("big", counted(10_000, 100, false)),
        ]);
        let filtered = filter("g", vec![small, big], &[], &found).unwrap();
        assert_eq!(ids(&filtered.kept), ["big"]);
    }

    #[rstest]
    fn each_deployment_is_checked_against_its_own_count() {
        let a = deployment("a", json!({"model": "openai/a"}), json!({}));
        let b = deployment("b", json!({"model": "openai/b"}), json!({}));
        let found = facts(&[
            ("a", counted(100, 101, false)),
            ("b", counted(100, 99, false)),
        ]);
        let filtered = filter("g", vec![a, b], &[], &found).unwrap();
        assert_eq!(ids(&filtered.kept), ["b"]);
    }

    #[rstest]
    #[case::exact_count_over_the_window(150, false, false)]
    #[case::estimate_within_the_margin(200, true, true)]
    #[case::estimate_past_the_margin(201, true, false)]
    fn only_an_estimate_within_the_margin_is_left_to_the_provider(
        #[case] tokens: i64,
        #[case] estimated: bool,
        #[case] kept: bool,
    ) {
        let only = deployment("only", json!({"model": "cohere/a"}), json!({}));
        let found = facts(&[("only", counted(100, tokens, estimated))]);
        let result = filter("g", vec![only], &[], &found);
        match result {
            Ok(filtered) => {
                assert!(kept, "expected a context window error");
                assert_eq!(ids(&filtered.kept), ["only"]);
                assert_eq!(
                    filtered.uncertain.get("only").map(String::as_str),
                    Some(format!("Model=cohere/a, Max Input Tokens=100, Got={tokens}").as_str())
                );
            }
            Err(error) => {
                assert!(!kept, "expected the deployment kept, got {error:?}");
                assert!(matches!(error, Error::ContextWindowExceeded { .. }));
            }
        }
    }

    #[rstest]
    fn an_uncertain_deployment_gives_way_to_one_the_prompt_fits() {
        let small = deployment("small", json!({"model": "cohere/a"}), json!({}));
        let big = deployment("big", json!({"model": "cohere/b"}), json!({}));
        let found = facts(&[
            ("small", counted(100, 150, true)),
            ("big", counted(1_000, 150, true)),
        ]);
        let filtered = filter("g", vec![small, big], &[], &found).unwrap();
        let settled = settle("g", filtered.kept, &filtered.uncertain, false).unwrap();
        assert_eq!(ids(&settled), ["big"]);
    }

    #[rstest]
    #[case::with_a_context_window_fallback(true)]
    #[case::without_one(false)]
    fn only_uncertain_deployments_left_go_to_a_fallback_or_the_provider(#[case] fallback: bool) {
        let one = deployment("one", json!({"model": "cohere/a"}), json!({}));
        let two = deployment("two", json!({"model": "cohere/b"}), json!({}));
        let found = facts(&[
            ("one", counted(100, 150, true)),
            ("two", counted(120, 150, true)),
        ]);
        let filtered = filter("g", vec![one, two], &[], &found).unwrap();
        let settled = settle("g", filtered.kept, &filtered.uncertain, fallback);
        if !fallback {
            assert_eq!(ids(&settled.unwrap()), ["one", "two"]);
            return;
        }
        let Err(Error::ContextWindowExceeded {
            model,
            detail,
            held_for_fallback: true,
        }) = settled
        else {
            panic!("expected a context window error, got {settled:?}");
        };
        assert_eq!(model, "g");
        assert_eq!(
            detail,
            "Model=cohere/a, Max Input Tokens=100, Got=150Model=cohere/b, Max Input Tokens=120, Got=150"
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
        let found = facts(&[("one", counted(5, 9, false)), ("two", counted(6, 8, false))]);
        let error = filter("g", vec![one, two], &[], &found).unwrap_err();
        let Error::ContextWindowExceeded {
            model,
            detail,
            held_for_fallback: false,
        } = error
        else {
            panic!("expected a context window error, got {error:?}");
        };
        assert_eq!(model, "g");
        assert_eq!(
            detail,
            "Model=openai/one, Max Input Tokens=5, Got=9Model=azure/base, Max Input Tokens=6, Got=8"
        );
    }

    #[rstest]
    fn without_input_nothing_is_checked() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[("small", counted(5, 100, false))]);
        found.has_input = false;
        assert_eq!(filter("g", vec![small], &[], &found).unwrap().kept.len(), 1);
    }

    #[rstest]
    fn a_prompt_that_may_not_be_counted_fits_every_window() {
        let small = deployment("small", json!({"model": "openai/a"}), json!({}));
        let mut found = facts(&[("small", counted(5, 100, false))]);
        found.countable = false;
        assert_eq!(filter("g", vec![small], &[], &found).unwrap().kept.len(), 1);
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
        let left = filter("g", vec![eu, none], &[], &found).unwrap();
        assert_eq!(ids(&left.kept), kept);
    }

    #[rstest]
    fn params_are_checked_only_on_deployments_inside_the_region() {
        let eu = deployment(
            "eu",
            json!({"model": "openai/old", "region_name": "eu"}),
            json!({}),
        );
        let ap = deployment(
            "ap",
            json!({"model": "openai/new", "region_name": "ap"}),
            json!({}),
        );
        let mut found = facts(&[(
            "eu",
            DeploymentFacts {
                lacks_params: true,
                ..DeploymentFacts::default()
            },
        )]);
        found.requested_params = vec!["response_format".to_owned()];
        found.allowed_model_region = Some(json!("eu"));
        let kept = filter("g", vec![eu, ap], &[], &found).unwrap();
        assert_eq!(ids(&kept.kept), ["eu"]);
    }

    #[rstest]
    fn a_group_where_every_deployment_rejects_the_param_is_unsupported() {
        let old = deployment("old", json!({"model": "openai/old"}), json!({}));
        let mut found = facts(&[(
            "old",
            DeploymentFacts {
                lacks_params: true,
                rejects_params: true,
                ..DeploymentFacts::default()
            },
        )]);
        found.requested_params = vec!["response_format".to_owned()];
        let error = filter("g", vec![old.clone()], &[old], &found).unwrap_err();
        assert!(
            matches!(error, Error::UnsupportedParams { params, .. } if params == ["response_format"])
        );
    }

    #[rstest]
    fn a_deployment_that_takes_the_param_wins_over_one_that_would_drop_it() {
        let old = deployment("old", json!({"model": "openai/old"}), json!({}));
        let new = deployment("new", json!({"model": "openai/new"}), json!({}));
        let mut found = facts(&[(
            "old",
            DeploymentFacts {
                lacks_params: true,
                ..DeploymentFacts::default()
            },
        )]);
        found.requested_params = vec!["response_format".to_owned()];
        let filtered = filter("g", vec![old, new], &[], &found).unwrap();
        assert_eq!(ids(&filtered.kept), ["new"]);
    }
}
