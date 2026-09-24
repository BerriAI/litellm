#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py::test_cost_calculator_with_response_cost_in_additional_headers
// mirrors: local_testing/test_completion_cost.py
use litellm_cost::error::CostError;

use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::completion_input::{CompletionInputRequest, ResponseKind};
use litellm_cost::completion_response::{
    BuiltInToolCostConfig, CompletionResponseCostRequest, CompletionTextInput,
    completion_cost_from_response, response_cost_calculator_from_response,
    response_time_ms_for_cost,
};
use litellm_cost::custom_pricing::{CustomPricing, CustomTokenRates};
use litellm_cost::model_selection::ModelSelectionRequest;
use litellm_cost::tool_call_cost_tracking::{DefaultToolRates, ResponseKind as ToolResponseKind};
use litellm_token_counter::{Error as TokenCounterError, TokenCounter, Tokenizer};
use rstest::rstest;
use serde_json::{Value, json};

struct CharacterTokenizer;

impl Tokenizer for CharacterTokenizer {
    fn count_tokens(&self, text: &str) -> Result<usize, TokenCounterError> {
        Ok(text.chars().count())
    }
}

fn request<'a>(
    response: Option<&'a Value>,
    model: Option<&'a str>,
    provider: Option<&'a str>,
    discount: &'a Value,
    margin: &'a Value,
) -> CompletionResponseCostRequest<'a> {
    CompletionResponseCostRequest {
        input: CompletionInputRequest {
            model_selection: ModelSelectionRequest {
                model,
                response,
                hidden_params: None,
                base_model: None,
                custom_pricing: false,
                provider,
                router_model_id: None,
                region_name: None,
            },
            call_type: None,
            response_kind: Some(ResponseKind::Completion),
            service_tier: None,
            optional_params: None,
        },
        fallback_usage: None,
        text_input: None,
        custom_cost: CustomPricing::NONE,
        replicate_rate_per_second: None,
        provider,
        region: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse().unwrap(),
        response_time_ms: None,
        prompt_characters: None,
        speech_prompt: None,
        completion_characters: None,
        transcription_duration_seconds: None,
        request_model: None,
        deployment_info: None,
        image_quality: None,
        image_size: None,
        image_count: None,
        built_in_tool_cost: 0.0,
        built_in_tool_config: None,
        additional_costs: &[],
        discount_config: discount,
        margin_config: margin,
        logging_details: None,
    }
}

#[rstest]
#[case(true, json!({"additional_headers": {"llm_provider-x-litellm-response-cost": "invalid"}}), Ok(0.0))]
#[case(false, json!({"additional_headers": {"llm_provider-x-litellm-response-cost": "0.5"}}), Ok(0.5))]
#[case(false, json!({"additional_headers": {"llm_provider-x-litellm-response-cost": 0.0}}), Ok(0.0))]
#[case(false, json!({"additional_headers": {"llm_provider-x-litellm-response-cost": "invalid"}}), Err(CostError::InvalidProviderCost))]
fn response_cost_calculator_short_circuits_cache_and_provider_cost(
    #[case] cache_hit: bool,
    #[case] hidden: Value,
    #[case] expected: Result<f64, CostError>,
) {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("unpriced"),
        Some("openai"),
        &empty,
        &empty,
    );
    let actual = response_cost_calculator_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..base.input.model_selection
                },
                ..base.input
            },
            ..base
        },
        cache_hit,
    );
    assert_eq!(actual, expected);
}

#[rstest]
fn response_cost_calculator_prices_response_without_provider_override() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/served".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let response = json!({
        "model": "served",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}
    });
    let hidden = json!({"additional_headers": {"llm_provider-x-litellm-response-cost": null}});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("requested"),
        Some("openai"),
        &empty,
        &empty,
    );
    let actual = response_cost_calculator_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..base.input.model_selection
                },
                ..base.input
            },
            ..base
        },
        false,
    );
    assert_eq!(actual, Ok(0.14));
}

#[rstest]
fn response_cost_infers_provider_for_specialized_pricing() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "xai/served".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let response = json!({
        "model": "xai/served",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 130,
            "completion_tokens_details": {"reasoning_tokens": 20}
        }
    });
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        request(Some(&response), Some("xai/served"), None, &empty, &empty),
    )
    .unwrap();
    assert!((result.cost.total - (100.0 * 0.01 + 30.0 * 0.02)).abs() < 1e-12);
}

#[rstest]
fn response_cost_rejects_a_bare_model_get_llm_provider_cannot_route() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "served".to_owned(),
        json!({"litellm_provider": "xai", "input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let response = json!({
        "model": "served",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    });
    let empty = json!({});
    assert_eq!(
        completion_cost_from_response(
            &catalog,
            request(Some(&response), Some("served"), None, &empty, &empty),
        )
        .map(|result| result.cost.total),
        Err(CostError::MissingProvider)
    );
}

#[rstest]
fn response_cost_infers_provider_from_priced_fallback_model() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "xai/served".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let response = json!({
        "model": "xai/served",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 130,
            "completion_tokens_details": {"reasoning_tokens": 20}
        }
    });
    let empty = json!({});
    let base = request(Some(&response), Some("missing"), None, &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    base_model: Some("missing"),
                    ..base.input.model_selection
                },
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert_eq!(result.model, "xai/served");
    assert!((result.cost.total - (100.0 * 0.01 + 30.0 * 0.02)).abs() < 1e-12);
}

#[rstest]
#[case(json!({"model": "model-router"}), None, true)]
#[case(json!({"model": "", "litellm_model_name": "model-router"}), None, true)]
#[case(json!({}), Some("model-router"), true)]
#[case(json!({"model": "model-router"}), Some("regular"), true)]
#[case(json!({}), Some("regular"), false)]
fn azure_ai_response_cost_bills_router_fee_once_from_request_or_hidden_model(
    #[case] hidden: Value,
    #[case] request_model: Option<&str>,
    #[case] has_router_fee: bool,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "azure_ai/served".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
        (
            "azure_ai/model_router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        ),
        (
            "azure_ai/model-router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        ),
    ]));
    let response = json!({
        "model": "azure_ai/served",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20}
    });
    let empty = json!({});
    let base = request(Some(&response), Some("served"), None, &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..base.input.model_selection
                },
                ..base.input
            },
            request_model,
            ..base
        },
    )
    .unwrap();
    let expected = 100.0 * 0.002 + 20.0 * 0.003 + if has_router_fee { 100.0 * 0.001 } else { 0.0 };
    assert!((result.cost.total - expected).abs() < 1e-12);
    assert!((result.cost.input - 100.0 * 0.002).abs() < 1e-12);
    assert!((result.cost.output - 20.0 * 0.003).abs() < 1e-12);
    assert_eq!(
        result.cost.additional,
        if has_router_fee { 0.1 } else { 0.0 }
    );
    assert_eq!(
        result
            .named_additional_costs
            .get("Azure Model Router Flat Cost"),
        has_router_fee.then_some(&0.1),
    );
}

#[rstest]
#[case(true, 0.1)]
#[case(false, 0.0)]
fn azure_ai_custom_token_pricing_keeps_router_fee_as_additional_cost(
    #[case] has_router_price: bool,
    #[case] expected_fee: f64,
) {
    let catalog = ModelInfoCatalog::new(
        [(
            "azure_ai/served".to_owned(),
            json!({"input_cost_per_token": 0.002}),
        )]
        .into_iter()
        .chain(has_router_price.then_some((
            "azure_ai/model_router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        )))
        .collect(),
    );
    let response = json!({
        "model": "azure_ai/served",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20}
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("served"),
        Some("azure_ai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            request_model: Some("model_router/deployment"),
            custom_cost: CustomPricing {
                token: Some(CustomTokenRates {
                    input: 0.004,
                    output: 0.005,
                    cache_read: None,
                    cache_creation: None,
                }),
                per_second: None,
            },
            ..base
        },
    )
    .unwrap();

    assert_eq!(result.cost.input, 0.4);
    assert_eq!(result.cost.output, 0.1);
    assert_eq!(result.cost.additional, expected_fee);
    assert!((result.cost.total - (0.5 + expected_fee)).abs() < 1e-12);
    assert_eq!(
        result
            .named_additional_costs
            .get("Azure Model Router Flat Cost"),
        (expected_fee > 0.0).then_some(&expected_fee),
    );
}

#[rstest]
fn completion_cost_counts_prompt_and_completion_without_response() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/plain".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let counter = TokenCounter::new(CharacterTokenizer);
    let empty = json!({});
    let base = request(None, Some("plain"), Some("openai"), &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            text_input: Some(CompletionTextInput {
                prompt: "hello",
                messages: None,
                completion: "yes",
                counter: &counter,
            }),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.11).abs() < 1e-12);
    assert_eq!(result.token_breakdown, None);
}

#[rstest]
#[case(None, 10.0)]
#[case(Some(3.0), 3.0)]
fn speech_response_cost_counts_prompt_characters_when_not_supplied(
    #[case] prompt_characters: Option<f64>,
    #[case] billable_characters: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/speech".to_owned(),
        json!({"input_cost_per_character": 0.01}),
    )]));
    let response = json!({});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("speech"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                response_kind: Some(ResponseKind::Speech),
                ..base.input
            },
            prompt_characters,
            speech_prompt: Some("hello world"),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - billable_characters * 0.01).abs() < 1e-12);
}

#[rstest]
fn completion_cost_counts_messages_before_prompt_without_response() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/plain".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let counter = TokenCounter::new(CharacterTokenizer);
    let messages = json!([{"role": "user", "content": "hello"}]);
    let empty = json!({});
    let base = request(None, Some("plain"), Some("openai"), &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            text_input: Some(CompletionTextInput {
                prompt: "ignored",
                messages: Some(&messages),
                completion: "yes",
                counter: &counter,
            }),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.21).abs() < 1e-12);
}

#[rstest]
fn invalid_messages_do_not_fall_back_to_prompt_pricing() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/plain".to_owned(),
        json!({"input_cost_per_token": 0.01}),
    )]));
    let counter = TokenCounter::new(CharacterTokenizer);
    let messages = json!({"role": "user", "content": "hello"});
    let empty = json!({});
    let base = request(None, Some("plain"), Some("openai"), &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            text_input: Some(CompletionTextInput {
                prompt: "cheap",
                messages: Some(&messages),
                completion: "",
                counter: &counter,
            }),
            ..base
        },
    );
    assert_eq!(result, Err(CostError::TokenCount));
}

#[rstest]
#[case(json!({"model": "plain"}))]
#[case(json!({"model": "plain", "usage": null}))]
fn response_without_usage_does_not_count_request_text(#[case] response: Value) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/plain".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let counter = TokenCounter::new(CharacterTokenizer);
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("plain"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            text_input: Some(CompletionTextInput {
                prompt: "hello",
                messages: None,
                completion: "yes",
                counter: &counter,
            }),
            ..base
        },
    )
    .unwrap();
    assert_eq!(result.cost.total, 0.0);
}

#[rstest]
fn custom_token_rates_override_catalog_and_provider_reported_cost() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({
        "model": "custom",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
            "cost": 99.0
        }
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("custom"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            custom_cost: CustomPricing {
                token: Some(CustomTokenRates {
                    input: 0.01,
                    output: 0.02,
                    cache_read: Some(0.001),
                    cache_creation: Some(0.03),
                }),
                per_second: Some(1.0),
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.164).abs() < 1e-12);
}

#[rstest]
fn custom_seconds_use_stamped_response_duration_without_catalog_price() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({
        "model": "custom",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "_response_ms": 2500.0
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("custom"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            custom_cost: CustomPricing {
                token: None,
                per_second: Some(0.08),
            },
            response_time_ms: Some(5000.0),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.2).abs() < 1e-12);
}

#[rstest]
fn custom_seconds_without_response_use_default_empty_text() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let empty = json!({});
    let base = request(None, Some("custom"), Some("openai"), &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            custom_cost: CustomPricing {
                token: None,
                per_second: Some(0.08),
            },
            response_time_ms: Some(2500.0),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.2).abs() < 1e-12);
}

#[rstest]
fn custom_cache_rates_match_python_anthropic_usage() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({
        "model": "custom",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2
        }
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("custom"),
        Some("anthropic"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            custom_cost: CustomPricing {
                token: Some(CustomTokenRates {
                    input: 0.01,
                    output: 0.02,
                    cache_read: Some(0.001),
                    cache_creation: Some(0.03),
                }),
                per_second: None,
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.284).abs() < 1e-12);
    let breakdown = result.token_breakdown.unwrap();
    assert!((breakdown.cache_read_cost - 4.0 * 0.001).abs() < 1e-12);
    assert!((breakdown.cache_creation_cost - 2.0 * 0.03).abs() < 1e-12);
    assert_eq!(breakdown.rates.unwrap().cache_read_input_token_cost, 0.001);
}

#[rstest]
fn unregistered_replicate_response_uses_time_and_skips_margin() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({
        "model": "replicate/unregistered",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}
    });
    let discount = json!({"replicate": 0.5});
    let margin = json!({"global": {"fixed_amount": 0.1}});
    let base = request(
        Some(&response),
        Some("replicate/unregistered"),
        Some("replicate"),
        &discount,
        &margin,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            response_time_ms: Some(2500.0),
            replicate_rate_per_second: Some(0.02),
            custom_cost: CustomPricing {
                token: Some(CustomTokenRates {
                    input: 1.0,
                    output: 1.0,
                    cache_read: None,
                    cache_creation: None,
                }),
                per_second: None,
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.05).abs() < 1e-12);
    assert_eq!(result.cost.margin_fixed_amount, 0.0);
}

#[rstest]
fn registered_replicate_model_uses_catalog_rates() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "replicate/mapped".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let response = json!({
        "model": "replicate/mapped",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("replicate/mapped"),
        Some("replicate"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            response_time_ms: Some(2500.0),
            replicate_rate_per_second: Some(0.02),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.14).abs() < 1e-12, "{result:?}");
}

#[rstest]
#[case("send_message", None, json!({"litellm_params": {"cost_per_query": 0.04}}), 0.04)]
#[case("asend_message", None, json!({"litellm_params": {"input_cost_per_token": 0.01}, "usage": {"prompt_tokens": 4}}), 0.04)]
#[case("call_mcp_tool", Some("tool"), json!({"mcp_tool_call_metadata": {"name": "search", "mcp_server_cost_info": {"tool_name_to_cost_per_query": {"search": 0.07}}}}), 0.07)]
fn metadata_priced_calls_use_logging_details_without_catalog_pricing(
    #[case] call_type: &str,
    #[case] model: Option<&str>,
    #[case] details: Value,
    #[case] expected: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let empty = json!({});
    let base = request(Some(&empty), model, None, &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some(call_type),
                ..base.input
            },
            logging_details: Some(&details),
            ..base
        },
    )
    .unwrap();
    assert_eq!(result.cost.total, expected);
}

#[rstest]
fn response_cost_prices_built_in_web_search_from_selected_model() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/served".to_owned(),
        json!({
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.0,
            "search_context_cost_per_query": {"search_context_size_medium": 0.02}
        }),
    )]));
    let response = json!({
        "model": "served",
        "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        "output": [{"type": "web_search_call"}, {"type": "web_search_call"}]
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("unpriced"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            built_in_tool_cost: 0.005,
            built_in_tool_config: Some(BuiltInToolCostConfig {
                response_kind: ToolResponseKind::Responses,
                params: &empty,
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
            }),
            ..base
        },
    )
    .unwrap();
    assert_eq!(result.model, "openai/served");
    assert!((result.cost.total - 0.145).abs() < 1e-12);
}

#[rstest]
#[case(json!({"_response_ms": 2000.0}), Some(5000.0), Some(2000.0))]
#[case(json!({"_response_ms": "not numeric"}), Some(5000.0), Some(5000.0))]
#[case(json!({}), None, None)]
fn response_time_prefers_numeric_response_stamp(
    #[case] response: Value,
    #[case] fallback: Option<f64>,
    #[case] expected: Option<f64>,
) {
    assert_eq!(
        response_time_ms_for_cost(Some(&response), fallback),
        expected
    );
}

#[rstest]
fn stamped_response_time_prices_wall_clock_model() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/timed".to_owned(),
        json!({"input_cost_per_second": 0.02}),
    )]));
    let response = json!({
        "model": "timed",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "_response_ms": 2000.0
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("timed"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            response_time_ms: Some(5000.0),
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.04).abs() < 1e-12);
}

#[rstest]
fn response_cost_falls_back_to_served_model_and_bills_served_tier() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/served".to_owned(),
        json!({
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
    let discount = json!({"anthropic": 0.1});
    let margin = json!({"global": 0.2});
    let auto = json!("auto");
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    base_model: Some("unpriced"),
                    ..request(
                        Some(&response),
                        Some("requested"),
                        Some("anthropic"),
                        &discount,
                        &margin,
                    )
                    .input
                    .model_selection
                },
                service_tier: Some(&auto),
                ..request(
                    Some(&response),
                    Some("requested"),
                    Some("anthropic"),
                    &discount,
                    &margin,
                )
                .input
            },
            built_in_tool_cost: 0.1,
            additional_costs: &[0.05],
            ..request(
                Some(&response),
                Some("requested"),
                Some("anthropic"),
                &discount,
                &margin,
            )
        },
    )
    .unwrap();
    assert_eq!(result.model, "served");
    assert_eq!(
        result.prepared.model_candidates[0].as_deref(),
        Some("anthropic/unpriced")
    );
    assert_eq!(result.prepared.service_tier.as_deref(), Some("priority"));
    assert!((result.cost.original - 0.65).abs() < 1e-12);
    assert!((result.cost.total - 0.702).abs() < 1e-12);
    let rates = result.token_breakdown.unwrap().rates.unwrap();
    assert_eq!(rates.input_cost_per_token, 0.03);
    assert_eq!(rates.output_cost_per_token, 0.04);
}

#[rstest]
fn search_response_uses_query_list_and_skips_tool_and_additional_charges() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "exa_ai/search".to_owned(),
        json!({"input_cost_per_query": 0.002}),
    )]));
    let response = json!({"model": "search"});
    let optional_params = json!({"query": ["a", "b", "c"]});
    let discount = json!({"exa_ai": 0.5});
    let margin = json!({"global": 0.1});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            custom_cost: CustomPricing {
                token: Some(CustomTokenRates {
                    input: 100.0,
                    output: 100.0,
                    cache_read: None,
                    cache_creation: None,
                }),
                per_second: None,
            },
            built_in_tool_cost: 1.0,
            additional_costs: &[2.0],
            input: CompletionInputRequest {
                call_type: Some("search"),
                optional_params: Some(&optional_params),
                ..request(
                    Some(&response),
                    Some("search"),
                    Some("exa_ai"),
                    &discount,
                    &margin,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("search"),
                Some("exa_ai"),
                &discount,
                &margin,
            )
        },
    )
    .unwrap();
    assert!((result.cost.original - 0.006).abs() < 1e-12);
    assert_eq!(result.cost.built_in_tools, 0.0);
    assert_eq!(result.cost.additional, 0.0);
    assert!((result.cost.total - 0.0033).abs() < 1e-12);
}

#[rstest]
#[case::empty_query_list_bills_nothing("exa_ai", "search", json!({"query": []}), 0.0)]
#[case::single_query_string("exa_ai", "search", json!({"query": "a"}), 0.5)]
#[case::bare_model_reads_the_provider_search_row("tavily", "tavily-search", json!({"query": ["a", "b"]}), 1.0)]
#[case::prefixed_model_keeps_its_own_row("exa_ai", "exa_ai/search", json!({}), 0.5)]
fn search_response_counts_queries_and_resolves_the_search_row_like_python(
    #[case] provider: &str,
    #[case] model: &str,
    #[case] optional_params: Value,
    #[case] expected: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "exa_ai/search".to_owned(),
            json!({"input_cost_per_query": 0.5}),
        ),
        (
            "tavily/search".to_owned(),
            json!({"input_cost_per_query": 0.5}),
        ),
    ]));
    let response = json!({"model": model});
    let empty = json!({});
    let base = request(Some(&response), Some(model), Some(provider), &empty, &empty);
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("search"),
                optional_params: Some(&optional_params),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - expected).abs() < 1e-12);
}

#[rstest]
#[case("vector_store_search", "vertex_ai/search_api", json!({}), 0.25)]
#[case("avector_store_search", "vertex_ai/search_api", json!({}), 0.25)]
#[case("vector_store_search", "vertex_ai/other", json!({"api_type": "search_api"}), 0.0)]
fn vector_store_response_uses_model_api_type_for_pricing(
    #[case] call_type: &str,
    #[case] model: &str,
    #[case] optional_params: Value,
    #[case] expected: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/search_api".to_owned(),
        json!({"input_cost_per_query": 0.25}),
    )]));
    let response = json!({"model": model});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some(model),
        Some("vertex_ai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some(call_type),
                optional_params: Some(&optional_params),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert_eq!(result.cost.total, expected);
}

#[rstest]
fn transcription_response_uses_hidden_duration_when_tokens_are_absent() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/transcribe".to_owned(),
        json!({"input_cost_per_second": 0.02}),
    )]));
    let response =
        json!({"model": "transcribe", "usage": {"prompt_tokens": 0, "completion_tokens": 0}});
    let hidden = json!({"audio_transcription_duration": 3.0});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("transcription"),
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..request(
                        Some(&response),
                        Some("transcribe"),
                        Some("openai"),
                        &empty,
                        &empty,
                    )
                    .input
                    .model_selection
                },
                ..request(
                    Some(&response),
                    Some("transcribe"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("transcribe"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.06).abs() < 1e-12);
}

#[rstest]
fn retrieve_batch_response_uses_batch_rates() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/batch".to_owned(),
        json!({
            "input_cost_per_token_batches": 0.01,
            "output_cost_per_token_batches": 0.02
        }),
    )]));
    let response = json!({
        "model": "batch",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("retrieve_batch"),
                ..request(
                    Some(&response),
                    Some("batch"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("batch"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.2).abs() < 1e-12);
}

#[rstest]
fn hidden_provider_selects_pricing_and_margin_after_model_fallback() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/served".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.0}),
    )]));
    let response =
        json!({"model": "served", "usage": {"prompt_tokens": 10, "completion_tokens": 0}});
    let hidden = json!({"custom_llm_provider": "anthropic"});
    let discount = json!({"anthropic": 0.1});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..request(
                        Some(&response),
                        Some("requested"),
                        Some("openai"),
                        &discount,
                        &empty,
                    )
                    .input
                    .model_selection
                },
                ..request(
                    Some(&response),
                    Some("requested"),
                    Some("openai"),
                    &discount,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("requested"),
                Some("openai"),
                &discount,
                &empty,
            )
        },
    )
    .unwrap();
    assert_eq!(result.model, "served");
    assert!((result.cost.total - 0.09).abs() < 1e-12);
}

#[rstest]
fn explicit_base_model_suppresses_regional_catalog_lookup() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/base".to_owned(),
            json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.0}),
        ),
        (
            "openai/eu/base".to_owned(),
            json!({"input_cost_per_token": 0.02, "output_cost_per_token": 0.0}),
        ),
    ]));
    let response = json!({"usage": {"prompt_tokens": 10, "completion_tokens": 0}});
    let hidden = json!({"region_name": "eu"});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    base_model: Some("base"),
                    hidden_params: Some(&hidden),
                    ..request(
                        Some(&response),
                        Some("requested"),
                        Some("openai"),
                        &empty,
                        &empty,
                    )
                    .input
                    .model_selection
                },
                ..request(
                    Some(&response),
                    Some("requested"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            region: Some("eu"),
            ..request(
                Some(&response),
                Some("requested"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert_eq!(result.model, "openai/base");
    assert!((result.cost.total - 0.1).abs() < 1e-12);
}

#[rstest]
fn image_response_uses_image_route_before_discount_and_margin() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "recraft/image".to_owned(),
        json!({"output_cost_per_image": 0.2}),
    )]));
    let response = json!({"data": [{}, {}]});
    let discount = json!({"recraft": 0.5});
    let margin = json!({"global": 0.5});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                response_kind: Some(ResponseKind::ImageGeneration),
                call_type: Some("image_generation"),
                ..request(
                    Some(&response),
                    Some("image"),
                    Some("recraft"),
                    &discount,
                    &margin,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("image"),
                Some("recraft"),
                &discount,
                &margin,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.4).abs() < 1e-12);
    assert_eq!(result.cost.discount_percent, 0.0);
    assert_eq!(result.cost.margin_percent, 0.0);
}

#[rstest]
#[case("image_generation")]
#[case("aimage_generation")]
fn azure_image_response_uses_dall_e_2_for_empty_model(#[case] call_type: &str) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "standard/1024-x-1024/dall-e-2".to_owned(),
        json!({"output_cost_per_image": 0.04}),
    )]));
    let response = json!({"data": [{}, {}]});
    let empty = json!({});
    let base = request(Some(&response), Some(""), Some("azure"), &empty, &empty);
    let actual = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                response_kind: Some(ResponseKind::ImageGeneration),
                call_type: Some(call_type),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert_eq!(actual.model, "azure/dall-e-2");
    assert_eq!(actual.cost.total, 2.0 * 0.04);
}

#[rstest]
fn video_response_prefers_provider_total_without_deployment_and_multiplies_custom_rate() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/video".to_owned(),
        json!({"output_cost_per_second": 0.02, "output_cost_per_second_1080p": 0.08}),
    )]));
    let response = json!({"usage": {
        "duration_seconds": 10.0,
        "video_resolution": " 1080P ",
        "video_count": 2,
        "provider_reported_cost_usd": 0.31
    }});
    let discount = json!({"openai": 0.5});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("video"),
        Some("openai"),
        &discount,
        &empty,
    );
    let reported = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("create_video"),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((reported.cost.total - 0.31).abs() < 1e-12);
    let deployment = json!({"output_cost_per_video_per_second": 0.05});
    let custom = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("video_edit"),
                model_selection: ModelSelectionRequest {
                    custom_pricing: true,
                    ..base.input.model_selection
                },
                ..base.input
            },
            deployment_info: Some(&deployment),
            ..base
        },
    )
    .unwrap();
    assert!((custom.cost.total - 1.0).abs() < 1e-12);
    assert_eq!(custom.cost.discount_percent, 0.0);
}

#[rstest]
fn video_status_poll_does_not_bill_response_time_as_generation_duration() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/video".to_owned(),
        json!({"mode": "video_generation", "output_cost_per_second": 0.2}),
    )]));
    let response = json!({"status": "completed"});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("video_retrieve"),
                ..request(
                    Some(&response),
                    Some("video"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            response_time_ms: Some(2000.0),
            ..request(
                Some(&response),
                Some("video"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert_eq!(result.cost.total, 0.0);
}

#[rstest]
fn realtime_response_prices_session_tokens_and_completed_transcription() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/session".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
        (
            "openai/asr".to_owned(),
            json!({"input_cost_per_second": 0.02}),
        ),
    ]));
    let response = json!({"results": [
        {"type": "session.created", "session": {"model": "session", "audio": {"input": {"transcription": {"model": "asr"}}}}},
        {"type": "response.done", "response": {"usage": {"input_tokens": 100, "output_tokens": 20}}},
        {"type": "conversation.item.input_audio_transcription.completed", "usage": {"type": "duration", "seconds": 2.0}},
        {"type": "conversation.item.input_audio_transcription.failed", "usage": {"type": "duration", "seconds": 100.0}}
    ]});
    let discount = json!({"openai": 0.5});
    let margin = json!({"global": {"fixed_amount": 0.1}});
    let base = request(
        Some(&response),
        Some("requested"),
        Some("openai"),
        &discount,
        &margin,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("_arealtime"),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.3).abs() < 1e-12);
    assert_eq!(result.cost.discount_percent, 0.0);
    assert_eq!(result.cost.margin_fixed_amount, 0.0);
}

#[rstest]
#[case::event_tiers_without_a_request_tier(None, None, 100.0 * 0.002 + 60.0 * 0.005)]
#[case::explicit_tier_overrides_every_event(Some(json!("priority")), None, 160.0 * 0.005)]
#[case::optional_params_tier_overrides_every_event(None, Some(json!({"service_tier": "priority"})), 160.0 * 0.005)]
#[case::auto_request_tier_keeps_event_tiers(Some(json!("auto")), None, 100.0 * 0.002 + 60.0 * 0.005)]
fn responses_websocket_prices_parts_at_the_requested_tier_first(
    #[case] service_tier: Option<Value>,
    #[case] optional_params: Option<Value>,
    #[case] expected_input: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 0.002,
            "output_cost_per_token": 0.0,
            "input_cost_per_token_priority": 0.005,
            "output_cost_per_token_priority": 0.0
        }),
    )]));
    let response = json!({"results": [
        {"type": "response.completed", "response": {"service_tier": "default", "usage": {"input_tokens": 100, "output_tokens": 0}}},
        {"type": "response.completed", "response": {"service_tier": "priority", "usage": {"input_tokens": 60, "output_tokens": 0}}}
    ]});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("model"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("_aresponses_websocket"),
                service_tier: service_tier.as_ref(),
                optional_params: optional_params.as_ref(),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.input - expected_input).abs() < 1e-12);
}

#[rstest]
fn responses_websocket_prices_each_tier_and_applies_fixed_margin_per_tier() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 0.002,
            "output_cost_per_token": 0.003,
            "input_cost_per_token_priority": 0.005,
            "output_cost_per_token_priority": 0.007
        }),
    )]));
    let response = json!({"results": [
        {"type": "response.completed", "response": {"service_tier": "default", "usage": {"input_tokens": 100, "output_tokens": 40}}},
        {"type": "response.incomplete", "response": {"service_tier": "priority", "usage": {"input_tokens": 60, "output_tokens": 10}}},
        {"type": "response.failed", "response": {"service_tier": "priority", "usage": {"input_tokens": 1000, "output_tokens": 1000}}}
    ]});
    let discount = json!({"openai": 0.1});
    let margin = json!({"global": {"fixed_amount": 0.1}});
    let base = request(
        Some(&response),
        Some("model"),
        Some("openai"),
        &discount,
        &margin,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("_aresponses_websocket"),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.original - 0.69).abs() < 1e-12);
    assert!((result.cost.input - 0.5).abs() < 1e-12);
    assert!((result.cost.output - 0.19).abs() < 1e-12);
    assert_eq!(result.cost.built_in_tools, 0.0);
    assert_eq!(result.cost.additional, 0.0);
    assert!((result.cost.total - 0.821).abs() < 1e-12);
    assert!((result.cost.margin_fixed_amount - 0.2).abs() < 1e-12);
    assert_eq!(result.cost.discount_percent, 0.1);
}

#[rstest]
fn responses_websocket_applies_custom_rates_to_each_tier() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({"results": [
        {"type": "response.completed", "response": {"service_tier": "default", "usage": {"input_tokens": 10, "output_tokens": 2}}},
        {"type": "response.completed", "response": {"service_tier": "priority", "usage": {"input_tokens": 4, "output_tokens": 3}}}
    ]});
    let empty = json!({});
    let margin = json!({"global": {"fixed_amount": 0.1}});
    let base = request(
        Some(&response),
        Some("custom"),
        Some("openai"),
        &empty,
        &margin,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("_aresponses_websocket"),
                ..base.input
            },
            custom_cost: CustomPricing {
                token: Some(CustomTokenRates {
                    input: 0.01,
                    output: 0.02,
                    cache_read: None,
                    cache_creation: None,
                }),
                per_second: None,
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.original - 0.24).abs() < 1e-12);
    assert!((result.cost.total - 0.44).abs() < 1e-12);
}

#[rstest]
fn empty_responses_websocket_applies_one_fixed_margin() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
    )]));
    let response = json!({"results": [{"type": "response.failed", "response": {"usage": null}}]});
    let empty = json!({});
    let margin = json!({"global": {"fixed_amount": 0.1}});
    let base = request(
        Some(&response),
        Some("model"),
        Some("openai"),
        &empty,
        &margin,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("_aresponses_websocket"),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.1).abs() < 1e-12);
}

#[rstest]
#[case::anthropic_messages("anthropic_messages")]
#[case::pass_through_endpoint("pass_through_endpoint")]
#[case::image_generation_without_an_image_response("image_generation")]
#[case::unknown_call_type("not_a_call_type")]
fn unlisted_call_types_fall_through_to_token_pricing_like_python(#[case] call_type: &str) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}),
    )]));
    let response = json!({"model": "model", "usage": {"prompt_tokens": 3, "completion_tokens": 4}});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("model"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some(call_type),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((result.cost.total - 11.0).abs() < 1e-12);
}

#[rstest]
fn rateless_custom_pricing_strips_provider_reported_cost_like_python() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "xai/grafling".to_string(),
        json!({"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6, "litellm_provider": "xai"}),
    )]));
    let response = json!({
        "model": "xai/grafling",
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 99.0}
    });
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("xai/grafling"),
        Some("xai"),
        &empty,
        &empty,
    );

    let recomputed = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    custom_pricing: true,
                    ..base.input.model_selection
                },
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((recomputed.cost.total - (1000.0 * 3e-6 + 500.0 * 15e-6)).abs() < 1e-12);

    let reported = completion_cost_from_response(&catalog, base).unwrap();
    assert!((reported.cost.total - 99.0).abs() < 1e-12);
}

#[rstest]
fn responses_websocket_prices_built_in_tools_from_each_tiers_usage() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 0.002,
            "output_cost_per_token": 0.003,
            "search_context_cost_per_query": {"search_context_size_medium": 0.02}
        }),
    )]));
    let response = json!({"results": [
        {"type": "response.completed", "response": {"service_tier": "default", "usage": {
            "input_tokens": 100, "output_tokens": 40,
            "input_token_details": {"web_search_requests": 2}
        }}},
        {"type": "response.incomplete", "response": {"service_tier": "priority", "usage": {
            "input_tokens": 60, "output_tokens": 10,
            "input_token_details": {"web_search_requests": 3}
        }}}
    ]});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("model"),
        Some("openai"),
        &empty,
        &empty,
    );
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("_aresponses_websocket"),
                ..base.input
            },
            built_in_tool_config: Some(BuiltInToolCostConfig {
                response_kind: ToolResponseKind::Other,
                params: &empty,
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
            }),
            ..base
        },
    )
    .unwrap();
    let tokens = 100.0 * 0.002 + 40.0 * 0.003 + 60.0 * 0.002 + 10.0 * 0.003;
    assert!((result.cost.built_in_tools - 2.0 * 0.02).abs() < 1e-12);
    assert!((result.cost.total - tokens - 2.0 * 0.02).abs() < 1e-12);
}
