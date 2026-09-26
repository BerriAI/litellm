#![allow(clippy::disallowed_types)]

// mirrors: unit/llms/groq/test_groq_cost_calculator.py::test_bills_searches_and_opens_together

use litellm_cost::groq_cost::cost_per_web_search_request;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn usage(value: Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

#[rstest]
#[case(json!({"web_search_requests": 2, "browser_open_requests": 3}), 0.023)]
#[case(json!({"web_search_requests": 2}), 0.02)]
#[case(json!({"browser_open_requests": 3}), 0.003)]
#[case(json!({}), 0.0)]
fn groq_prices_executed_browser_actions_independently(
    #[case] server_tool_use: Value,
    #[case] expected: f64,
) {
    let usage = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "server_tool_use": server_tool_use
    }));
    let model_info = json!({
        "search_context_cost_per_query": {"search_context_size_medium": 0.01}
    });
    assert!((cost_per_web_search_request(&usage, &model_info, 0.001) - expected).abs() < 1e-12);
}

#[rstest]
fn groq_without_server_tool_use_has_no_tool_charge() {
    let usage = usage(json!({"prompt_tokens": 10, "completion_tokens": 2}));
    assert_eq!(
        cost_per_web_search_request(
            &usage,
            &json!({"search_context_cost_per_query": {"search_context_size_medium": 0.01}}),
            0.001,
        ),
        0.0
    );
}
