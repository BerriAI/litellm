#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_completion_cost_extracts_service_tier_from_response
// mirrors: test_litellm/llms/gemini/test_cost_calculator.py::test_map_traffic_type_to_service_tier

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::completion_input::{
    CompletionInputRequest, ResponseKind, infer_call_type, map_traffic_type_to_service_tier,
    select_service_tier,
};
use litellm_cost::model_selection::ModelSelectionRequest;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case("ON_DEMAND_PRIORITY", Some("priority"))]
#[case("on_demand_flex", Some("flex"))]
#[case("BATCH", Some("flex"))]
#[case("FLEX", Some("flex"))]
#[case("ON_DEMAND", None)]
#[case("unknown", None)]
fn traffic_type_maps_only_billable_tiers(#[case] raw: &str, #[case] expected: Option<&str>) {
    assert_eq!(map_traffic_type_to_service_tier(Some(raw)), expected);
}

#[rstest]
fn request_auto_defers_to_served_usage_tier() {
    let explicit = json!("auto");
    let optional = json!({"service_tier": "flex"});
    let response = json!({"service_tier": {"name": "auto"}, "usage": {"service_tier": "priority"}});
    assert_eq!(
        select_service_tier(Some(&explicit), Some(&optional), Some(&response), None),
        Some("priority".to_owned())
    );
}

#[rstest]
#[case(json!("priority"), "priority")]
#[case(json!({"name": "auto"}), "flex")]
#[case(json!("AUTO"), "flex")]
fn request_tier_precedes_response_when_billable(#[case] requested: Value, #[case] expected: &str) {
    let response = json!({"service_tier": "flex", "usage": {"service_tier": "priority"}});
    assert_eq!(
        select_service_tier(Some(&requested), None, Some(&response), None),
        Some(expected.to_owned())
    );
}

#[rstest]
fn traffic_type_is_last_tier_fallback() {
    let response = json!({"usage": {"service_tier": "auto"}});
    let hidden = json!({"provider_specific_fields": {"traffic_type": "ON_DEMAND_FLEX"}});
    assert_eq!(
        select_service_tier(None, None, Some(&response), Some(&hidden)),
        Some("flex".to_owned())
    );
    let served = json!({"usage": {"service_tier": "priority"}});
    assert_eq!(
        select_service_tier(None, None, Some(&served), Some(&hidden)),
        Some("priority".to_owned())
    );
    assert_eq!(select_service_tier(None, None, None, Some(&hidden)), None);
}

#[rstest]
#[case(Some("batch"), Some(ResponseKind::Completion), Some("batch"))]
#[case(None, Some(ResponseKind::Embedding), Some("embedding"))]
#[case(None, Some(ResponseKind::ImageGeneration), Some("image_generation"))]
#[case(None, None, None)]
fn call_type_prefers_explicit_value(
    #[case] explicit: Option<&str>,
    #[case] kind: Option<ResponseKind>,
    #[case] expected: Option<&str>,
) {
    assert_eq!(infer_call_type(explicit, kind), expected);
}

#[rstest]
fn prepared_input_prices_served_priority_rate() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/served".to_owned(),
        json!({
            "litellm_provider": "anthropic",
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "input_cost_per_token_priority": 0.03,
            "output_cost_per_token_priority": 0.04
        }),
    )]));
    let response = json!({
        "model": "served",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "service_tier": "priority"}
    });
    let request_tier = json!("auto");
    let prepared = catalog
        .prepare_completion_input(CompletionInputRequest {
            model_selection: ModelSelectionRequest {
                model: Some("requested"),
                response: Some(&response),
                hidden_params: None,
                base_model: None,
                custom_pricing: false,
                provider: Some("anthropic"),
                router_model_id: None,
                region_name: None,
            },
            call_type: None,
            response_kind: Some(ResponseKind::Completion),
            service_tier: Some(&request_tier),
            optional_params: None,
        })
        .unwrap();
    assert_eq!(prepared.call_type, "completion");
    assert_eq!(
        prepared.model_candidates[0].as_deref(),
        Some("anthropic/served")
    );
    assert_eq!(prepared.service_tier.as_deref(), Some("priority"));
    let (prompt, output) = litellm_cost::cost_calculator::cost_per_token(
        &catalog,
        ModelCostRequest {
            model: prepared.model_candidates[0].as_deref().unwrap(),
            provider: Some("anthropic"),
            region: None,
            usage: prepared.usage.as_ref().unwrap(),
            service_tier: prepared.service_tier.as_deref(),
            data_residency: None,
            vertex_location: None,
            at: Timestamp::from_second(0).unwrap(),
            response_time_ms: None,
        },
    )
    .unwrap();
    assert!((prompt - 0.3).abs() < 1e-10);
    assert!((output - 0.2).abs() < 1e-10);
}

#[rstest]
fn non_string_response_tier_defers_to_the_served_usage_tier() {
    let response =
        json!({"service_tier": {"name": "priority"}, "usage": {"service_tier": "priority"}});
    assert_eq!(
        select_service_tier(None, None, Some(&response), None),
        Some("priority".to_owned())
    );
}

#[rstest]
#[case(json!({"name": "priority"}))]
#[case(json!(5))]
#[case(json!(true))]
fn non_string_usage_tier_falls_back_to_standard(#[case] usage_tier: Value) {
    let response = json!({"service_tier": {"name": "auto"}, "usage": {"service_tier": usage_tier}});
    assert_eq!(select_service_tier(None, None, Some(&response), None), None);
}
