//! What `Router::reach` reports for a name, from `router_settings` as a config writes them.

use litellm_router_core::{Router, RouterConfig};
use litellm_router_types::RouterSettings;
use rstest::rstest;
use serde_json::{Value, json};

/// The groups `claude`, `backup`, `last` and `other`, one deployment each, and one deployment
/// under the pattern `openai/*`. A case expects each group reached, in order, marked `unserved`
/// when no deployment serves it, then `+` the setting behind each of its host steps.
fn router(settings: Value) -> Router {
    let settings: RouterSettings = serde_json::from_value(settings).unwrap();
    let (config, dynamic) = RouterConfig::from_settings(&settings).unwrap();
    let model_list = ["claude", "backup", "last", "other", "openai/*"]
        .into_iter()
        .map(|group| {
            serde_json::from_value(json!({
                "model_name": group,
                "litellm_params": {"model": format!("anthropic/{group}")},
                "model_info": {"id": format!("{group}-1")},
            }))
            .unwrap()
        })
        .collect();
    Router::new(
        RouterConfig {
            model_list,
            ..config
        },
        dynamic,
    )
    .unwrap()
}

#[rstest]
#[case::no_fallbacks(json!({}), "claude", &["claude"])]
#[case::a_fallback(
    json!({"fallbacks": [{"claude": ["backup"]}]}),
    "claude",
    &["claude", "backup"]
)]
#[case::a_fallback_of_a_fallback(
    json!({"fallbacks": [{"claude": ["last"]}, {"last": ["backup"]}]}),
    "claude",
    &["claude", "last", "backup"]
)]
#[case::a_context_window_fallback(
    json!({"context_window_fallbacks": [{"claude": ["backup"]}]}),
    "claude",
    &["claude", "backup"]
)]
#[case::a_content_policy_fallback(
    json!({"content_policy_fallbacks": [{"claude": ["backup"]}]}),
    "claude",
    &["claude", "backup"]
)]
#[case::a_default_fallback(
    json!({"default_fallbacks": ["backup"]}),
    "other",
    &["other", "backup"]
)]
#[case::a_fallback_of_every_group(
    json!({"fallbacks": [{"*": ["backup"]}]}),
    "other",
    &["other", "backup"]
)]
#[case::its_own_chain_in_front_of_the_one_of_every_group(
    json!({"fallbacks": [{"claude": ["other"]}, {"*": ["backup"]}]}),
    "claude",
    &["claude", "other"]
)]
#[case::a_list_of_bare_names_is_every_groups_chain(
    json!({"fallbacks": ["backup"]}),
    "claude",
    &["claude", "backup"]
)]
#[case::a_fallback_to_an_alias(
    json!({"fallbacks": [{"claude": ["spare"]}], "model_group_alias": {"spare": "backup"}}),
    "claude",
    &["claude", "backup"]
)]
#[case::an_alias(
    json!({"model_group_alias": {"sonnet": {"model": "claude", "hidden": true}}}),
    "sonnet",
    &["claude"]
)]
#[case::a_chain_keyed_by_the_alias_asked_for(
    json!({"fallbacks": [{"sonnet": ["backup"]}], "model_group_alias": {"sonnet": "claude"}}),
    "sonnet",
    &["claude", "backup"]
)]
#[case::a_chain_keyed_by_an_alias_nobody_asked_for(
    json!({"fallbacks": [{"sonnet": ["backup"]}], "model_group_alias": {"sonnet": "claude"}}),
    "claude",
    &["claude"]
)]
#[case::a_fallback_to_no_group(
    json!({"fallbacks": [{"claude": ["nowhere"]}]}),
    "claude",
    &["claude", "nowhere unserved"]
)]
#[case::a_fallback_only_a_pattern_serves(
    json!({"fallbacks": [{"claude": ["openai/gpt"]}]}),
    "claude",
    &["claude", "openai/gpt unserved"]
)]
#[case::a_fallback_written_as_a_dict(
    json!({"fallbacks": [{"claude": [{"model": "backup", "temperature": 0}]}]}),
    "claude",
    &["claude +fallbacks", "backup +fallbacks"]
)]
#[case::a_context_window_fallback_written_as_a_dict(
    json!({"context_window_fallbacks": [{"claude": [{"model": "backup"}]}]}),
    "claude",
    &["claude +context_window_fallbacks", "backup +context_window_fallbacks"]
)]
#[case::a_strategy_the_core_asks_the_host_for(
    json!({"routing_strategy": "least-busy", "fallbacks": [{"claude": ["backup"]}]}),
    "claude",
    &["claude +routing_strategy", "backup +routing_strategy"]
)]
#[case::simple_shuffle(json!({"routing_strategy": "simple-shuffle"}), "claude", &["claude"])]
#[case::a_routing_group_with_its_own_strategy(
    json!({
        "routing_groups": [
            {"group_name": "pool", "models": ["backup"], "routing_strategy": "least-busy"},
        ],
        "fallbacks": [{"claude": ["backup"]}],
    }),
    "claude",
    &["claude", "backup +routing_groups"]
)]
#[case::a_priority_routing_group(
    json!({
        "routing_groups": [
            {"group_name": "pool", "models": ["claude"], "routing_strategy": "priority"},
        ],
    }),
    "claude",
    &["claude"]
)]
#[case::pre_call_checks_and_health_check_routing(
    json!({"enable_pre_call_checks": true, "enable_health_check_routing": true}),
    "claude",
    &["claude +enable_pre_call_checks +enable_health_check_routing"]
)]
fn a_name_reaches_its_group_then_each_group_its_fallbacks_reach(
    #[case] settings: Value,
    #[case] name: &str,
    #[case] expected: &[&str],
) {
    let reached = router(settings).reach(name).unwrap();

    let reported: Vec<String> = reached
        .iter()
        .map(|reached| {
            let unserved = if reached.served { "" } else { " unserved" };
            let steps: String = reached
                .steps
                .iter()
                .map(|step| format!(" +{}", step.setting()))
                .collect();
            format!("{}{unserved}{steps}", reached.group)
        })
        .collect();
    assert_eq!(reported, expected);
}

#[rstest]
#[case::no_group(json!({}), "gpt")]
#[case::a_name_only_a_pattern_serves(json!({}), "openai/gpt")]
#[case::a_deployment_id(json!({}), "claude-1")]
#[case::a_routing_group(
    json!({
        "routing_groups": [
            {"group_name": "pool", "models": ["claude"], "routing_strategy": "simple-shuffle"},
        ],
    }),
    "pool"
)]
#[case::an_alias_of_no_group(json!({"model_group_alias": {"sonnet": "gpt"}}), "sonnet")]
#[case::an_alias_without_a_model(json!({"model_group_alias": {"sonnet": {"hidden": true}}}), "sonnet")]
#[case::a_name_only_the_default_fallback_serves(json!({"default_fallbacks": ["backup"]}), "gpt")]
fn a_name_that_is_no_model_group_reaches_nothing(#[case] settings: Value, #[case] name: &str) {
    assert_eq!(router(settings).reach(name), None);
}
