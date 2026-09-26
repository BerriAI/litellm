#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_tool_call_cost_tracking.py::test_get_cost_for_built_in_tools_web_search

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::cost_calculator::{BuiltInToolCharge, CompletionCostRequest};
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

fn usage(value: &Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

#[rstest]
#[case("per_query", 0.04)]
#[case("per_prompt", 0.03)]
fn get_cost_for_built_in_tools_adds_gemini_search_and_maps(
    #[case] billing_unit: &str,
    #[case] expected: f64,
) {
    let usage = usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_tokens_details": {"web_search_requests": 2, "google_maps_grounding_requests": 1}
    }));
    let model_info = json!({
        "web_search_billing_unit": billing_unit,
        "search_context_cost_per_query": {"search_context_size_medium": 0.01},
        "google_maps_grounding_cost_per_query": 0.02
    });
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"choices": []}),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider: Some("gemini"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&model_info),
    );
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
fn get_cost_for_built_in_tools_prefers_reported_responses_search_count() {
    let response = json!({
        "output": [{"type": "web_search_call"}, {"type": "web_search_call"}],
        "tool_usage": {"web_search": {"num_requests": 1}}
    });
    let actual = priced(
        BuiltInToolCostRequest {
            response: &response,
            response_kind: ResponseKind::Responses,
            usage: None,
            provider: Some("openai"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.02}})),
    );
    assert_eq!(actual, 0.02);
}

#[rstest]
fn get_cost_for_built_in_tools_reads_anthropic_raw_web_search_usage() {
    let response = json!({"usage": {"server_tool_use": {"web_search_requests": 2}}});
    let actual = priced(
        BuiltInToolCostRequest {
            response: &response,
            response_kind: ResponseKind::Anthropic,
            usage: None,
            provider: Some("anthropic"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.02}})),
    );
    assert_eq!(actual, 0.04);
}

#[rstest]
#[case(Some(0.3), 0.0)]
#[case(None, 0.03)]
fn get_cost_for_built_in_tools_uses_xai_reported_cost_or_search_calls(
    #[case] reported: Option<f64>,
    #[case] expected: f64,
) {
    let usage = usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cost": reported,
        "server_side_tool_usage_details": {"web_search_calls": 2}
    }));
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"choices": []}),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider: Some("xai"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.015}})),
    );
    assert_eq!(actual, expected);
}

#[rstest]
fn get_cost_for_built_in_tools_prices_groq_search_and_browser_open() {
    let usage = usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "server_tool_use": {"web_search_requests": 2, "browser_open_requests": 3}
    }));
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"choices": []}),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider: Some("groq"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.01}})),
    );
    assert!((actual - 0.023).abs() < 1e-12);
}

#[rstest]
fn get_cost_for_built_in_tools_prices_file_search_before_assistant_features() {
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"output": [{"type": "file_search_call"}]}),
            response_kind: ResponseKind::Responses,
            usage: None,
            provider: Some("azure"),
            params: &json!({
                "file_search": {"type": "file_search", "storage_gb": 1.5, "days": 10},
                "code_interpreter_sessions": 3
            }),
            defaults: defaults(),
        },
        Some(&json!({"file_search_cost_per_gb_per_day": 0.4})),
    );
    assert_eq!(actual, 6.0);
}

#[rstest]
fn get_cost_for_built_in_tools_adds_azure_assistant_features() {
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"output": []}),
            response_kind: ResponseKind::Responses,
            usage: None,
            provider: Some("azure"),
            params: &json!({
                "vector_store_usage": {"storage_gb": 2.0, "days": 5.0},
                "computer_use_usage": {"input_tokens": 1000, "output_tokens": 500},
                "code_interpreter_sessions": 3
            }),
            defaults: defaults(),
        },
        None,
    );
    assert!((actual - 11.09).abs() < 1e-12);
}

#[rstest]
fn model_info_catalog_completion_cost_dispatches_built_in_tools() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "search_context_cost_per_query": {"search_context_size_medium": 0.01}
        }),
    )]));
    let usage = usage(&json!({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}));
    let response = json!({"output": [{"type": "web_search_call"}, {"type": "web_search_call"}]});
    let params = json!({});
    let discount = json!({});
    let margin = json!({});
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let actual = litellm_cost::cost_calculator::completion_cost(
        &catalog,
        CompletionCostRequest {
            token: ModelCostRequest {
                model: "model",
                provider: Some("openai"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at,
                response_time_ms: None,
            },
            built_in_tools: BuiltInToolCharge::FromResponse(BuiltInToolCostRequest {
                response: &response,
                response_kind: ResponseKind::Responses,
                usage: Some(&usage),
                provider: Some("openai"),
                params: &params,
                defaults: defaults(),
            }),
            additional_costs: &[],
            discount_config: &discount,
            margin_config: &margin,
        },
    )
    .unwrap();
    assert!((actual.total - (100.0 * 2e-6 + 50.0 * 4e-6 + 2.0 * 0.01)).abs() < 1e-12);
}

fn gemini_catalog() -> ModelInfoCatalog {
    ModelInfoCatalog::new(HashMap::from([(
        "gemini/gemini-x".to_owned(),
        json!({
            "litellm_provider": "gemini",
            "web_search_billing_unit": "per_query",
            "search_context_cost_per_query": {"search_context_size_medium": 0.014},
            "google_maps_grounding_cost_per_query": 0.0125
        }),
    )]))
}

#[rstest]
#[case::no_provider(None)]
#[case::provider_the_entry_does_not_belong_to(Some("openai"))]
fn get_cost_for_built_in_tools_adopts_the_provider_of_the_resolved_entry(
    #[case] provider: Option<&str>,
) {
    let usage = usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_tokens_details": {"web_search_requests": 3, "google_maps_grounding_requests": 2}
    }));
    let actual = get_cost_for_built_in_tools(
        &gemini_catalog(),
        "gemini/gemini-x",
        None,
        BuiltInToolCostRequest {
            response: &json!({"choices": []}),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider,
            params: &json!({}),
            defaults: defaults(),
        },
    );
    assert!((actual - (3.0 * 0.014 + 2.0 * 0.0125)).abs() < 1e-12);
}

fn cited_chat_response() -> Value {
    json!({"choices": [{"message": {"annotations": [{"type": "url_citation"}]}}]})
}

#[rstest]
fn anthropic_usage_without_a_search_count_bills_nothing_instead_of_the_per_call_price() {
    let usage = usage(&json!({"prompt_tokens": 10, "completion_tokens": 2}));
    let actual = priced(
        BuiltInToolCostRequest {
            response: &cited_chat_response(),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider: Some("anthropic"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.01}})),
    );
    assert_eq!(actual, 0.0);
}

#[rstest]
#[case::with_usage(true, 0.0)]
#[case::without_usage(false, 0.01)]
fn perplexity_search_is_free_only_when_usage_reaches_the_router(
    #[case] with_usage: bool,
    #[case] expected: f64,
) {
    let usage = usage(&json!({"prompt_tokens": 10, "completion_tokens": 2}));
    let actual = priced(
        BuiltInToolCostRequest {
            response: &cited_chat_response(),
            response_kind: ResponseKind::Chat,
            usage: with_usage.then_some(&usage),
            provider: Some("perplexity"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.01}})),
    );
    assert_eq!(actual, expected);
}

#[rstest]
#[case::per_prompt(json!("per_prompt"), 0.01)]
#[case::unset(json!(null), 0.01)]
#[case::empty(json!(""), 0.01)]
#[case::per_query(json!("per_query"), 0.03)]
#[case::unrecognized(json!("per_request"), 0.03)]
fn gemini_bills_one_search_only_for_per_prompt_units(
    #[case] billing_unit: Value,
    #[case] expected: f64,
) {
    let usage = usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_tokens_details": {"web_search_requests": 3}
    }));
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"choices": []}),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider: Some("gemini"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({
            "web_search_billing_unit": billing_unit,
            "search_context_cost_per_query": {"search_context_size_medium": 0.01}
        })),
    );
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
#[case::string_count(json!("3"), 0.03)]
#[case::integral_float_count(json!(2.0), 0.02)]
#[case::null_count_is_not_a_search(json!(null), 0.0)]
fn raw_anthropic_search_counts_are_coerced_like_pydantic(
    #[case] requests: Value,
    #[case] expected: f64,
) {
    let response = json!({"usage": {"server_tool_use": {"web_search_requests": requests}}});
    let actual = priced(
        BuiltInToolCostRequest {
            response: &response,
            response_kind: ResponseKind::Anthropic,
            usage: None,
            provider: Some("anthropic"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.01}})),
    );
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
#[case::integer(json!(1), 0.015)]
#[case::boolean_counts_as_one_call(json!(true), 0.015)]
#[case::float_is_not_an_int(json!(1.0), 0.0)]
fn xai_server_side_search_calls_follow_python_isinstance_int(
    #[case] calls: Value,
    #[case] expected: f64,
) {
    let usage = usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "server_side_tool_usage_details": {"web_search_calls": calls}
    }));
    let actual = priced(
        BuiltInToolCostRequest {
            response: &json!({"choices": []}),
            response_kind: ResponseKind::Chat,
            usage: Some(&usage),
            provider: Some("xai"),
            params: &json!({}),
            defaults: defaults(),
        },
        Some(&json!({"search_context_cost_per_query": {"search_context_size_medium": 0.015}})),
    );
    assert!((actual - expected).abs() < 1e-12);
}
