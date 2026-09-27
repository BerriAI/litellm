#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/gemini/test_cost_calculator.py

use std::collections::BTreeMap;

use litellm_cost::gemini_cost::{
    cost_per_google_maps_grounding_request, cost_per_web_search_request,
    google_maps_grounding_requests,
};
use litellm_cost::responses_usage::{ChatUsage, PromptTokenDetails};
use rstest::rstest;
use serde_json::{Value, json};

fn usage(
    web_search_requests: Option<u64>,
    maps_requests: Option<u64>,
    server_search_requests: Option<u64>,
) -> ChatUsage {
    ChatUsage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        prompt_tokens_details: (web_search_requests.is_some() || maps_requests.is_some())
            .then_some(PromptTokenDetails {
                web_search_requests,
                google_maps_grounding_requests: maps_requests,
                ..PromptTokenDetails::default()
            }),
        completion_tokens_details: None,
        cost: None,
        extra: server_search_requests.map_or_else(BTreeMap::new, |requests| {
            BTreeMap::from([(
                "server_tool_use".to_owned(),
                json!({"web_search_requests": requests}),
            )])
        }),
    }
}

#[rstest]
#[case(json!({"web_search_billing_unit": "per_query", "search_context_cost_per_query": {"search_context_size_medium": 0.014}}), 3, 0.042)]
#[case(json!({"search_context_cost_per_query": {"search_context_size_medium": 0.014}}), 3, 0.014)]
#[case(json!({"web_search_billing_unit": "per_query", "search_context_cost_per_query": {"search_context_size_medium": 0.014}}), 0, 0.0)]
fn cost_per_web_search_request_uses_configured_billing_unit(
    #[case] model_info: Value,
    #[case] requests: u64,
    #[case] expected: f64,
) {
    let actual = cost_per_web_search_request(&usage(Some(requests), None, None), &model_info);
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
#[case(None, Some(3), 3)]
#[case(Some(2), Some(5), 2)]
#[case(Some(0), Some(5), 5)]
fn cost_per_web_search_request_uses_server_tool_fallback_only_when_prompt_count_is_zero(
    #[case] prompt_requests: Option<u64>,
    #[case] server_requests: Option<u64>,
    #[case] expected_requests: u64,
) {
    let pricing = json!({
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {"search_context_size_medium": 0.01}
    });
    let actual =
        cost_per_web_search_request(&usage(prompt_requests, None, server_requests), &pricing);
    assert!((actual - expected_requests as f64 * 0.01).abs() < 1e-12);
}

#[rstest]
#[case(None)]
#[case(Some(0))]
#[case(Some(3))]
fn google_maps_grounding_requests_keeps_absent_and_zero_distinct(#[case] requests: Option<u64>) {
    assert_eq!(
        google_maps_grounding_requests(Some(&usage(None, requests, None))),
        requests
    );
    assert_eq!(google_maps_grounding_requests(None), None);
}

#[rstest]
#[case(json!({"web_search_billing_unit": "per_query", "google_maps_grounding_cost_per_query": 0.014}), 3, 0.042)]
#[case(json!({"google_maps_grounding_cost_per_query": 0.025}), 3, 0.025)]
#[case(json!({"web_search_billing_unit": "per_query", "google_maps_grounding_cost_per_query": 0.014}), 0, 0.0)]
fn cost_per_google_maps_grounding_request_uses_configured_billing_unit(
    #[case] model_info: Value,
    #[case] requests: u64,
    #[case] expected: f64,
) {
    let actual =
        cost_per_google_maps_grounding_request(&usage(None, Some(requests), None), &model_info);
    assert!((actual - expected).abs() < 1e-12);
}
