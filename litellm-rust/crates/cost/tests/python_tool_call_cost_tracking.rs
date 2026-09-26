#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_tool_call_cost_tracking.py::test_web_search_cost_low

use litellm_cost::tool_call_cost_tracking::{
    DefaultToolRates, ResponseKind, chat_completion_response_includes_annotations,
    count_web_search_calls, extract_file_search_params, extract_token_counts,
    get_cost_for_code_interpreter, get_cost_for_computer_use, get_cost_for_file_search,
    get_cost_for_vector_store, get_cost_for_web_search, response_object_includes_file_search_call,
    response_object_includes_web_search_call,
};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(json!({"choices": [{"message": {"annotations": []}}]}), false)]
#[case(json!({"choices": [{"message": {}}, {"message": {"annotations": [{"type": "unknown"}]}}]}), true)]
#[case(json!({"choices": [{"message": {"annotations": null}}]}), false)]
fn chat_completion_response_includes_any_annotation(
    #[case] response: serde_json::Value,
    #[case] expected: bool,
) {
    assert_eq!(
        chat_completion_response_includes_annotations(&response),
        expected
    );
}

fn defaults() -> DefaultToolRates {
    DefaultToolRates {
        file_search_per_call: 0.25,
        azure_file_search_per_gb_day: 0.1,
        azure_vector_store_per_gb_day: 0.2,
        azure_computer_input_per_1k_tokens: 3.0,
        azure_computer_output_per_1k_tokens: 12.0,
        code_interpreter_per_session: Some(0.03),
        xai_web_search_per_call: 0.005,
        groq_browser_open_per_call: 0.001,
    }
}

#[rstest]
#[case("low", 0.01)]
#[case("medium", 0.02)]
#[case("high", 0.03)]
#[case("unknown", 0.02)]
fn get_cost_for_web_search_selects_context_rate(#[case] size: &str, #[case] expected: f64) {
    let model_info = json!({"search_context_cost_per_query": {
        "search_context_size_low": 0.01,
        "search_context_size_medium": 0.02,
        "search_context_size_high": 0.03
    }});
    assert_eq!(
        get_cost_for_web_search(
            Some(&json!({"search_context_size": size})),
            Some(&model_info)
        ),
        expected
    );
    assert_eq!(get_cost_for_web_search(None, Some(&model_info)), 0.02);
    assert_eq!(get_cost_for_web_search(None, None), 0.0);
}

#[rstest]
fn get_cost_for_file_search_uses_storage_or_call_rate() {
    let search = json!({"type": "file_search"});
    let azure_info = json!({"file_search_cost_per_gb_per_day": 0.4});
    let other_info = json!({"file_search_cost_per_1k_calls": 0.7});
    assert_eq!(
        get_cost_for_file_search(
            Some(&search),
            Some("azure"),
            Some(&azure_info),
            Some(1.5),
            Some(10.0),
            defaults()
        ),
        6.0
    );
    assert_eq!(
        get_cost_for_file_search(
            Some(&search),
            Some("azure"),
            None,
            Some(1.5),
            Some(10.0),
            defaults()
        ),
        1.5
    );
    assert_eq!(
        get_cost_for_file_search(
            Some(&search),
            Some("openai"),
            Some(&other_info),
            None,
            None,
            defaults()
        ),
        0.7
    );
    assert_eq!(
        get_cost_for_file_search(
            Some(&search),
            Some("azure"),
            Some(&other_info),
            Some(1.5),
            Some(10.0),
            defaults()
        ),
        0.7
    );
    assert_eq!(
        get_cost_for_file_search(None, Some("azure"), None, Some(1.0), Some(1.0), defaults()),
        0.0
    );
}

#[rstest]
fn get_cost_for_vector_store_prices_model_override_and_azure_default() {
    let usage = json!({"storage_gb": 2.0, "days": 5.0});
    assert_eq!(
        get_cost_for_vector_store(Some(&usage), Some("azure"), None, defaults()),
        2.0
    );
    assert_eq!(
        get_cost_for_vector_store(
            Some(&usage),
            Some("openai"),
            Some(&json!({"vector_store_cost_per_gb_per_day": 0.4})),
            defaults()
        ),
        4.0
    );
    assert_eq!(
        get_cost_for_vector_store(Some(&usage), Some("openai"), None, defaults()),
        0.0
    );
}

#[rstest]
fn get_cost_for_computer_use_prices_azure_and_model_override() {
    assert_eq!(
        get_cost_for_computer_use(Some(1000), Some(500), Some("azure"), None, defaults()),
        9.0
    );
    assert_eq!(
        get_cost_for_computer_use(
            Some(1000),
            Some(500),
            Some("azure"),
            Some(&json!({"computer_use_input_cost_per_1k_tokens": 5.0,
                "computer_use_output_cost_per_1k_tokens": 10.0})),
            defaults()
        ),
        10.0
    );
    assert_eq!(
        get_cost_for_computer_use(Some(1000), Some(500), Some("openai"), None, defaults()),
        0.0
    );
}

#[rstest]
fn get_cost_for_code_interpreter_uses_model_then_catalog_rate() {
    assert_eq!(
        get_cost_for_code_interpreter(
            Some(3),
            Some(&json!({"code_interpreter_cost_per_session": 0.5})),
            defaults()
        ),
        1.5
    );
    assert_eq!(
        get_cost_for_code_interpreter(Some(3), None, defaults()),
        0.09
    );
    assert_eq!(get_cost_for_code_interpreter(None, None, defaults()), 0.0);
}

#[rstest]
fn extract_params_converts_python_inputs() {
    assert_eq!(
        extract_file_search_params(&json!({"storage_gb": "1.5", "days": "10"})),
        (Some(1.5), Some(10.0))
    );
    assert_eq!(
        extract_token_counts(&json!({"input_tokens": 12.8, "output_tokens": "4"})),
        (Some(12), Some(4))
    );
}

#[rstest]
fn response_detection_uses_explicit_api_shape() {
    let chat = json!({"choices": [{"message": {"annotations": [{"type": "url_citation"}]}}]});
    let responses = json!({"output": [{"type": "file_search_call"}, {"type": "web_search_call"}]});
    assert!(response_object_includes_web_search_call(
        &chat,
        ResponseKind::Chat,
        None
    ));
    assert!(response_object_includes_file_search_call(
        &responses,
        ResponseKind::Responses
    ));
    assert!(!response_object_includes_file_search_call(
        &chat,
        ResponseKind::Chat
    ));
    assert!(response_object_includes_web_search_call(
        &json!({}),
        ResponseKind::Other,
        Some(&json!({"server_tool_use": {"web_search_requests": 0}}))
    ));
}

#[rstest]
fn count_web_search_calls_prefers_reported_billable_requests() {
    let response = json!({
        "tool_usage": {"web_search": {"num_requests": 1}},
        "output": [{"type": "web_search_call"}, {"type": "web_search_call"}]
    });
    assert_eq!(
        count_web_search_calls(&response, ResponseKind::Responses),
        1
    );
    assert_eq!(
        count_web_search_calls(
            &json!({"tool_usage": {"web_search": {"num_requests": 0}}}),
            ResponseKind::Responses
        ),
        0
    );
    assert_eq!(
        count_web_search_calls(
            &json!({"output": [{"type": "web_search_call"}, {"type": "web_search_call"}]}),
            ResponseKind::Responses
        ),
        2
    );
    assert_eq!(
        count_web_search_calls(&json!({"output": []}), ResponseKind::Responses),
        1
    );
    assert_eq!(count_web_search_calls(&response, ResponseKind::Chat), 1);
}
