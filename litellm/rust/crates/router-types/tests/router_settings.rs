use std::collections::BTreeSet;

use litellm_router_types::{OptionalPreCallCheck, RouterSettings};
use rstest::rstest;
use serde_json::{Value, json};
use strum::IntoEnumIterator;

fn named() -> Vec<OptionalPreCallCheck> {
    OptionalPreCallCheck::iter()
        .filter(|check| !matches!(check, OptionalPreCallCheck::Other(_)))
        .collect()
}

fn checks(value: Value) -> serde_json::Result<Option<Vec<OptionalPreCallCheck>>> {
    serde_json::from_value::<RouterSettings>(json!({ "optional_pre_call_checks": value }))
        .map(|settings| settings.optional_pre_call_checks)
}

/// The names in the `OptionalPreCallChecks` `Literal` of `litellm/types/router.py`.
fn python_names() -> BTreeSet<String> {
    let source = include_str!("../../../../types/router.py");
    let (_, after) = source
        .split_once("OptionalPreCallChecks = list[")
        .expect("router.py defines OptionalPreCallChecks");
    let (literal, _) = after.split_once("]\n]").expect("the Literal closes");
    literal
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect()
}

#[rstest]
#[case::prompt_caching("prompt_caching", OptionalPreCallCheck::PromptCaching)]
#[case::router_budget_limiting(
    "router_budget_limiting",
    OptionalPreCallCheck::RouterBudgetLimiting
)]
#[case::responses_api_deployment_check(
    "responses_api_deployment_check",
    OptionalPreCallCheck::ResponsesApiDeploymentCheck
)]
#[case::deployment_affinity("deployment_affinity", OptionalPreCallCheck::DeploymentAffinity)]
#[case::session_affinity("session_affinity", OptionalPreCallCheck::SessionAffinity)]
#[case::forward_client_headers_by_model_group(
    "forward_client_headers_by_model_group",
    OptionalPreCallCheck::ForwardClientHeadersByModelGroup
)]
#[case::enforce_model_rate_limits(
    "enforce_model_rate_limits",
    OptionalPreCallCheck::EnforceModelRateLimits
)]
#[case::encrypted_content_affinity(
    "encrypted_content_affinity",
    OptionalPreCallCheck::EncryptedContentAffinity
)]
#[case::an_unknown_name_is_kept("not_a_check", OptionalPreCallCheck::Other("not_a_check".into()))]
#[case::an_empty_name_is_kept("", OptionalPreCallCheck::Other(String::new()))]
fn each_name_reads_as_its_check_and_writes_back_the_same(
    #[case] name: &str,
    #[case] expected: OptionalPreCallCheck,
) {
    assert_eq!(checks(json!([name])).unwrap(), Some(vec![expected.clone()]));
    assert_eq!(expected.to_string(), name);
}

#[rstest]
#[case::a_number(json!([1]))]
#[case::an_object(json!([{"name": "prompt_caching"}]))]
#[case::a_bare_name(json!("prompt_caching"))]
fn a_check_that_is_not_a_name_is_rejected(#[case] value: Value) {
    assert!(checks(value).is_err());
}

#[rstest]
fn the_named_checks_are_the_ones_python_lists() {
    let named: BTreeSet<String> = named().iter().map(ToString::to_string).collect();
    assert_eq!(named, python_names());
}

/// The names in `VALID_MODEL_GROUP_AFFINITY_FLAGS` of `deployment_affinity_check.py`.
fn python_affinity_flags() -> Vec<String> {
    let source =
        include_str!("../../../../router_utils/pre_call_checks/deployment_affinity_check.py");
    let (_, after) = source
        .split_once("VALID_MODEL_GROUP_AFFINITY_FLAGS: Final = frozenset(")
        .expect("deployment_affinity_check.py defines VALID_MODEL_GROUP_AFFINITY_FLAGS");
    let (set, _) = after.split_once(')').expect("the frozenset closes");
    set.split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect()
}

#[rstest]
fn each_affinity_flag_python_accepts_reads_as_a_named_check() {
    let flags = python_affinity_flags();
    let named = named();
    assert!(!flags.is_empty());
    let settings: RouterSettings =
        serde_json::from_value(json!({ "model_group_affinity_config": { "gpt": flags } })).unwrap();
    let read = &settings.model_group_affinity_config.unwrap()["gpt"];
    assert_eq!(read.len(), flags.len());
    for (check, flag) in read.iter().zip(&flags) {
        assert!(named.contains(check), "{flag} read as {check:?}");
        assert_eq!(check.to_string(), *flag);
    }
}

#[rstest]
#[case::an_unknown_flag(json!({ "gpt": ["not_a_flag"] }), Some(OptionalPreCallCheck::Other("not_a_flag".into())))]
#[case::a_number(json!({ "gpt": [1] }), None)]
fn an_affinity_flag_python_does_not_know_is_kept_and_a_non_name_is_rejected(
    #[case] config: Value,
    #[case] expected: Option<OptionalPreCallCheck>,
) {
    let read =
        serde_json::from_value::<RouterSettings>(json!({ "model_group_affinity_config": config }))
            .ok()
            .map(|settings| settings.model_group_affinity_config.unwrap()["gpt"].clone());
    assert_eq!(read, expected.map(|check| vec![check]));
}
