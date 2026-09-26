#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py::test_cost_per_token_duplicate_openai_prefix_matches_model_cost
// mirrors: test_litellm/test_utils.py::test_check_provider_match_azure_ai_allows_openai_and_azure
use litellm_cost::cost_calculator::{cost_per_token, cost_per_token_for_call};
use litellm_cost::error::CostError;

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::model_info::check_provider_match;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

fn rates(input: f64, output: f64) -> serde_json::Value {
    json!({"input_cost_per_token": input, "output_cost_per_token": output})
}

fn catalog() -> ModelInfoCatalog {
    ModelInfoCatalog::new(HashMap::from([
        ("openai/model".to_owned(), rates(1e-6, 2e-6)),
        ("openai/openai/model".to_owned(), rates(9e-6, 9e-6)),
        ("bedrock_mantle/model".to_owned(), rates(3e-6, 4e-6)),
        (
            "bedrock_mantle/us-gov-west-1/model".to_owned(),
            rates(5e-6, 6e-6),
        ),
        ("bare-model".to_owned(), rates(7e-6, 8e-6)),
    ]))
}

fn usage() -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 50}}))
        .unwrap()
        .unwrap()
}

fn request<'a>(
    model: &'a str,
    provider: Option<&'a str>,
    region: Option<&'a str>,
    usage: &'a litellm_cost::responses_usage::ChatUsage,
) -> ModelCostRequest<'a> {
    ModelCostRequest {
        model,
        provider,
        region,
        usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse().unwrap(),
        response_time_ms: None,
    }
}

#[rstest]
#[case("openai/openai/model", Some("openai"), None, Some("openai/model"))]
#[case("model", Some("openai"), None, Some("openai/model"))]
#[case("openai/model", None, None, Some("openai/model"))]
#[case(
    "bedrock_mantle/model",
    Some("bedrock_mantle"),
    Some("us-gov-west-1"),
    Some("bedrock_mantle/us-gov-west-1/model")
)]
#[case(
    "bedrock_mantle/model",
    Some("bedrock_mantle"),
    Some("missing"),
    Some("bedrock_mantle/model")
)]
#[case(
    "bedrock_mantle/bare-model",
    Some("bedrock_mantle"),
    None,
    Some("bare-model")
)]
#[case("missing", Some("openai"), None, None)]
fn cost_per_token_resolves_model_key_in_python_order(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] region: Option<&str>,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        catalog()
            .select_model_info(model, provider, region)
            .map(|selected| selected.key.into_owned())
            .as_deref(),
        expected
    );
}

#[rstest]
fn cost_per_token_uses_the_selected_regional_prices() {
    let usage = usage();
    let (input, output) = cost_per_token_for_call(
        &catalog(),
        request(
            "bedrock_mantle/model",
            Some("bedrock_mantle"),
            Some("us-gov-west-1"),
            &usage,
        ),
        litellm_cost::catalog::CostCall::Token {
            call_type: "completion",
            prompt_characters: None,
            completion_characters: None,
            request_model: None,
        },
    )
    .unwrap();
    assert!((input - 100.0 * 5e-6).abs() < 1e-12);
    assert!((output - 50.0 * 6e-6).abs() < 1e-12);
}

#[rstest]
fn cost_per_token_reports_an_unmapped_model() {
    assert_eq!(
        cost_per_token(
            &catalog(),
            request("missing", Some("openai"), None, &usage())
        ),
        Err(CostError::ModelNotFound)
    );
}

#[rstest]
fn model_info_catalog_prices_selected_model_with_off_peak_and_region() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/model".to_owned(),
            json!({
                "input_cost_per_token": 2e-6,
                "output_cost_per_token": 4e-6,
                "regional_processing_uplift_multiplier_eu": 1.2,
                "off_peak_pricing": {
                    "hours_utc": "16:30-00:30",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 2e-6,
                    "cache_read_input_token_cost": 0.25e-6
                }
            }),
        ),
        (
            "openai/openai/model".to_owned(),
            json!({"input_cost_per_token": 9e-6, "output_cost_per_token": 9e-6}),
        ),
    ]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20}
    }}))
    .unwrap()
    .unwrap();
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let actual = cost_per_token(
        &catalog,
        ModelCostRequest {
            model: "openai/openai/model",
            provider: Some("openai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: Some("eu"),
            vertex_location: None,
            at,
            response_time_ms: None,
        },
    )
    .unwrap();
    assert!((actual.0 - (80.0 * 1e-6 + 20.0 * 0.25e-6) * 1.2).abs() < 1e-12);
    assert!((actual.1 - 50.0 * 2e-6 * 1.2).abs() < 1e-12);
}

#[rstest]
fn model_info_catalog_applies_xai_inclusive_threshold_policy() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "xai/model".to_owned(),
        json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "input_cost_per_token_above_128k_tokens": 5e-6,
            "output_cost_per_token_above_128k_tokens": 7e-6
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 128_000,
        "completion_tokens": 10,
        "total_tokens": 128_010
    }}))
    .unwrap()
    .unwrap();
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let actual = cost_per_token(
        &catalog,
        ModelCostRequest {
            model: "model",
            provider: Some("xai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: None,
        },
    )
    .unwrap();
    assert!((actual.0 - 128_000.0 * 5e-6).abs() < 1e-12);
    assert!((actual.1 - 10.0 * 7e-6).abs() < 1e-12);
}

#[rstest]
#[case::no_requested_provider(Some("gemini"), None, true)]
#[case::empty_requested_provider(Some("gemini"), Some(""), true)]
#[case::entry_without_provider(None, Some("openai"), true)]
#[case::same_provider(Some("openai"), Some("openai"), true)]
#[case::different_provider(Some("gemini"), Some("openai"), false)]
#[case::vertex_family(Some("vertex_ai-language-models"), Some("vertex_ai"), true)]
#[case::vertex_family_is_one_way(Some("vertex_ai"), Some("vertex_ai-language-models"), false)]
#[case::fireworks_family(Some("fireworks_ai-embedding-models"), Some("fireworks_ai"), true)]
#[case::fireworks_request_rejects_other_providers(Some("openai"), Some("fireworks_ai"), false)]
#[case::fireworks_entry_needs_a_fireworks_request(
    Some("fireworks_ai-embedding-models"),
    Some("together_ai"),
    false
)]
#[case::bedrock_family(Some("bedrock_converse"), Some("bedrock"), true)]
#[case::bedrock_request_rejects_other_providers(Some("openai"), Some("bedrock"), false)]
#[case::bedrock_entry_needs_a_bedrock_request(Some("bedrock"), Some("sagemaker"), false)]
#[case::litellm_proxy_matches_anything(Some("anthropic"), Some("litellm_proxy"), true)]
#[case::azure_ai_falls_back_to_azure(Some("azure"), Some("azure_ai"), true)]
#[case::azure_ai_falls_back_to_openai(Some("openai"), Some("azure_ai"), true)]
#[case::azure_does_not_fall_back_to_azure_ai(Some("azure_ai"), Some("azure"), false)]
#[case::github_reuses_any_entry(Some("openai"), Some("github"), true)]
fn check_provider_match_follows_python(
    #[case] entry_provider: Option<&str>,
    #[case] requested: Option<&str>,
    #[case] expected: bool,
) {
    let model_info = json!({"litellm_provider": entry_provider});
    assert_eq!(check_provider_match(&model_info, requested), expected);
}

#[rstest]
#[case::matching_provider(Some("gemini"), Some("gemini/gemini-x"))]
#[case::no_provider(None, Some("gemini/gemini-x"))]
#[case::mismatched_provider(Some("openai"), None)]
fn select_model_key_skips_entries_owned_by_another_provider(
    #[case] provider: Option<&str>,
    #[case] expected: Option<&str>,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "gemini/gemini-x".to_owned(),
        json!({"litellm_provider": "gemini", "input_cost_per_token": 1e-6}),
    )]));
    assert_eq!(
        catalog
            .select_model_info("gemini/gemini-x", provider, None)
            .map(|selected| selected.key.into_owned())
            .as_deref(),
        expected
    );
}

#[rstest]
fn select_model_key_lets_vertex_ai_beta_use_vertex_entries() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/gemini-x".to_owned(),
        json!({"litellm_provider": "vertex_ai-language-models"}),
    )]));
    assert_eq!(
        catalog
            .select_model_info("vertex_ai/gemini-x", Some("vertex_ai_beta"), None)
            .map(|selected| selected.key.into_owned())
            .as_deref(),
        Some("vertex_ai/gemini-x")
    );
}
