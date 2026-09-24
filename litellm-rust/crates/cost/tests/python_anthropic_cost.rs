#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_anthropic_geo_and_fast_multipliers_compose
// mirrors: test_litellm/llms/anthropic/test_cost_calculation_dict_safety.py::test_get_cost_for_anthropic_web_search_with_dict_server_tool_use

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::anthropic_cost::{
    fast_speed_multiplier, get_anthropic_web_search_requests_from_response,
    get_cost_for_anthropic_web_search, get_web_search_requests, get_web_search_requests_from_usage,
};
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::tool_call_cost_tracking::{DefaultToolRates, ResponseKind};
use litellm_cost::tool_cost_dispatch::{BuiltInToolCostRequest, get_cost_for_built_in_tools};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn priced(request: BuiltInToolCostRequest<'_>, model_info: Option<&Value>) -> f64 {
    let provider = request.provider.expect("tests price a provider");
    let catalog = ModelInfoCatalog::new(
        model_info
            .map(|info| (format!("{provider}/model"), info.clone()))
            .into_iter()
            .collect(),
    );
    get_cost_for_built_in_tools(&catalog, "model", None, request)
}

fn usage(value: &Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

#[rstest]
fn anthropic_cost_per_token_composes_served_tier_cache_geo_and_fast_once() {
    let model_info = json!({
        "input_cost_per_token": 0.003,
        "output_cost_per_token": 0.015,
        "cache_read_input_token_cost": 0.0003,
        "input_cost_per_token_priority": 0.006,
        "output_cost_per_token_priority": 0.030,
        "cache_read_input_token_cost_priority": 0.0006,
        "provider_specific_entry": {"fast": 2.0, "us": 1.1}
    });
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/model".to_owned(),
        model_info.clone(),
    )]));
    let usage = usage(&json!({
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 200},
        "speed": "fast",
        "inference_geo": "us"
    }));
    let at: Timestamp = "2026-09-22T12:00:00Z".parse().unwrap();
    let (prompt, completion) = litellm_cost::cost_calculator::cost_per_token(
        &catalog,
        ModelCostRequest {
            model: "model",
            provider: Some("anthropic"),
            region: None,
            usage: &usage,
            service_tier: Some("priority"),
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: None,
        },
    )
    .unwrap();
    let expected_prompt = (800.0 * 0.006 + 200.0 * 0.0006) * 1.1 * 2.0;
    let expected_completion = 500.0 * 0.030 * 1.1 * 2.0;
    assert!((prompt - expected_prompt).abs() < 1e-12);
    assert!((completion - expected_completion).abs() < 1e-12);
    assert_eq!(fast_speed_multiplier(&model_info, &usage), 2.0);
}

#[rstest]
fn anthropic_standard_speed_preserves_geo_without_fast_multiplier() {
    let info = json!({"provider_specific_entry": {"fast": 2.0, "us": 1.1}});
    let usage = usage(&json!({"prompt_tokens": 100, "speed": "standard", "inference_geo": "us"}));
    assert_eq!(fast_speed_multiplier(&info, &usage), 1.0);
}

#[rstest]
#[case(json!({"usage": {"server_tool_use": {"web_search_requests": 4}}}), Some(4))]
#[case(json!({"usage": {"server_tool_use": {}}}), None)]
#[case(json!({"usage": null}), None)]
fn get_anthropic_web_search_requests_from_response_reads_nested_usage(
    #[case] response: Value,
    #[case] expected: Option<i64>,
) {
    assert_eq!(
        get_anthropic_web_search_requests_from_response(&response),
        expected
    );
}

#[rstest]
#[case(json!({"web_search_requests": 0}), Some(0))]
#[case(json!({"web_search_requests": 3}), Some(3))]
#[case(json!({}), None)]
#[case(json!({"web_search_requests": "3"}), Some(3))]
#[case(json!({"web_search_requests": 2.0}), Some(2))]
#[case(json!({"web_search_requests": 2.5}), None)]
#[case(json!({"web_search_requests": null}), None)]
fn get_web_search_requests_reads_serialized_server_tool_use(
    #[case] server_tool_use: Value,
    #[case] expected: Option<i64>,
) {
    assert_eq!(get_web_search_requests(Some(&server_tool_use)), expected);
    assert_eq!(get_web_search_requests(None), None);
}

#[rstest]
fn get_cost_for_anthropic_web_search_prices_server_tool_requests() {
    let model_info =
        json!({"search_context_cost_per_query": {"search_context_size_medium": 0.003}});
    let usage = usage(&json!({"prompt_tokens": 10, "server_tool_use": {"web_search_requests": 3}}));
    assert_eq!(get_web_search_requests_from_usage(&usage), Some(3));
    assert!(
        (get_cost_for_anthropic_web_search(Some(&model_info), Some(&usage)) - 0.009).abs() < 1e-12
    );
    assert_eq!(get_cost_for_anthropic_web_search(None, Some(&usage)), 0.0);
    assert_eq!(
        get_cost_for_anthropic_web_search(Some(&model_info), None),
        0.0
    );
    let dispatched = priced(
        BuiltInToolCostRequest {
            response: &json!({}),
            response_kind: ResponseKind::Anthropic,
            usage: Some(&usage),
            provider: Some("anthropic"),
            params: &json!({}),
            defaults: DefaultToolRates {
                file_search_per_call: 0.0,
                azure_file_search_per_gb_day: 0.0,
                azure_vector_store_per_gb_day: 0.0,
                azure_computer_input_per_1k_tokens: 0.0,
                azure_computer_output_per_1k_tokens: 0.0,
                code_interpreter_per_session: None,
                xai_web_search_per_call: 0.0,
                groq_browser_open_per_call: 0.0,
            },
        },
        Some(&model_info),
    );
    assert!((dispatched - 0.009).abs() < 1e-12);
}
