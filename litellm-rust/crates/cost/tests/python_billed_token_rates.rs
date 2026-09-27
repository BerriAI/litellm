#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_get_token_type_cost_breakdown_reflects_off_peak_reasoning_and_cache_creation_rates

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::billed_token_rates::{
    BilledRatesRequest, calculate_token_type_cost_breakdown, get_billed_token_rates,
    get_token_type_cost_breakdown,
};
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::custom_pricing::CustomTokenRates;
use litellm_cost::responses_usage::{ChatUsage, CompletionTokenDetails, PromptTokenDetails};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn at(value: &str) -> Timestamp {
    value.parse().unwrap()
}

#[rstest]
#[case("2026-01-01T18:00Z", 5e-7, 5e-7)]
#[case("2026-01-01T12:00Z", 4e-6, 1.25e-6)]
fn get_token_type_cost_breakdown_uses_off_peak_reasoning_and_cache_write_rates(
    #[case] billed_at: &str,
    #[case] reasoning_rate: f64,
    #[case] write_rate: f64,
) {
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_reasoning_token": 4e-6,
        "cache_read_input_token_cost": 1e-7,
        "cache_creation_input_token_cost": 1.25e-6,
        "off_peak_pricing": {
            "hours_utc": "16:30-00:30",
            "output_cost_per_reasoning_token": 5e-7,
            "cache_creation_input_token_cost": 5e-7
        }
    });
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 80,
        "total_tokens": 1080,
        "prompt_tokens_details": {"cached_tokens": 100, "cache_creation_tokens": 400, "text_tokens": 500},
        "completion_tokens_details": {"reasoning_tokens": 30, "text_tokens": 50}
    }}))
    .unwrap()
    .unwrap();
    let breakdown = calculate_token_type_cost_breakdown(BilledRatesRequest {
        model_info: Some(&model_info),
        usage: &usage,
        provider: Some("openai"),
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: at(billed_at),
        custom_cost_per_token: None,
    });
    assert!((breakdown.reasoning_cost - 30.0 * reasoning_rate).abs() < 1e-12);
    assert!((breakdown.cache_creation_cost - 400.0 * write_rate).abs() < 1e-12);
    assert!((breakdown.cache_read_cost - 100.0 * 1e-7).abs() < 1e-12);
    assert_eq!(
        breakdown.rates.unwrap().output_cost_per_reasoning_token,
        reasoning_rate
    );
}

#[rstest]
fn get_token_type_cost_breakdown_uses_flat_custom_rates_without_model_metadata() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 800, "cache_creation_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 200}
    }}))
    .unwrap()
    .unwrap();
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let request = ModelCostRequest {
        model: "unmapped",
        provider: Some("openai"),
        region: None,
        usage: &usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: at("2026-01-01T18:00Z"),
        response_time_ms: None,
    };
    let custom = CustomTokenRates {
        input: 1e-6,
        output: 2e-6,
        cache_read: Some(1e-7),
        cache_creation: None,
    };
    let breakdown = get_token_type_cost_breakdown(&catalog, request, Some(custom));
    assert!((breakdown.cache_read_cost - 800.0 * 1e-7).abs() < 1e-12);
    assert!((breakdown.cache_creation_cost - 100.0 * 1e-6).abs() < 1e-12);
    assert!((breakdown.reasoning_cost - 200.0 * 2e-6).abs() < 1e-12);
    assert_eq!(
        get_billed_token_rates(&catalog, request, Some(custom)),
        breakdown.rates
    );
    assert_eq!(get_billed_token_rates(&catalog, request, None), None);
}

#[rstest]
fn get_billed_token_rates_follows_selected_threshold_tier() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
            "input_cost_per_token_above_200k_tokens": 6e-6,
            "output_cost_per_token_above_200k_tokens": 3e-5,
            "cache_read_input_token_cost_above_200k_tokens": 6e-7,
            "cache_creation_input_token_cost_above_200k_tokens": 7.5e-6
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 250_000,
        "completion_tokens": 1_000,
        "total_tokens": 251_000,
        "prompt_tokens_details": {"cached_tokens": 200_000, "cache_creation_tokens": 10_000},
        "completion_tokens_details": {"reasoning_tokens": 200}
    }}))
    .unwrap()
    .unwrap();
    let request = ModelCostRequest {
        model: "model",
        provider: Some("openai"),
        region: None,
        usage: &usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: at("2026-01-01T18:00Z"),
        response_time_ms: None,
    };
    let rates = get_billed_token_rates(&catalog, request, None).unwrap();
    let breakdown = get_token_type_cost_breakdown(&catalog, request, None);
    assert_eq!(rates.input_cost_per_token, 6e-6);
    assert_eq!(rates.output_cost_per_token, 3e-5);
    assert_eq!(rates.cache_read_input_token_cost, 6e-7);
    assert_eq!(rates.cache_creation_input_token_cost, 7.5e-6);
    assert_eq!(breakdown.rates, Some(rates));
    assert!((breakdown.cache_read_cost - 200_000.0 * 6e-7).abs() < 1e-12);
    assert!((breakdown.cache_creation_cost - 10_000.0 * 7.5e-6).abs() < 1e-12);
    assert!((breakdown.reasoning_cost - 200.0 * 3e-5).abs() < 1e-12);
}

#[rstest]
fn get_token_type_cost_breakdown_scales_cache_audio_and_ttl_with_region() {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "output_cost_per_reasoning_token": 8e-6,
        "cache_read_input_token_cost": 0.5e-6,
        "cache_read_input_audio_token_cost": 0.25e-6,
        "cache_creation_input_token_cost": 3e-6,
        "cache_creation_input_token_cost_above_1hr": 4e-6,
        "regional_processing_uplift_multiplier_eu": 1.2
    });
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {
            "cached_tokens": 200,
            "cached_tokens_details": {"audio_tokens": 80},
            "cache_creation_tokens": 100,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": 40,
                "ephemeral_1h_input_tokens": 60
            }
        },
        "completion_tokens_details": {"reasoning_tokens": 20}
    }}))
    .unwrap()
    .unwrap();
    let breakdown = calculate_token_type_cost_breakdown(BilledRatesRequest {
        model_info: Some(&model_info),
        usage: &usage,
        provider: Some("openai"),
        service_tier: None,
        data_residency: Some("eu"),
        vertex_location: None,
        at: at("2026-01-01T18:00Z"),
        custom_cost_per_token: None,
    });
    assert!((breakdown.cache_read_cost - (120.0 * 0.5e-6 + 80.0 * 0.25e-6) * 1.2).abs() < 1e-12);
    assert!((breakdown.cache_creation_cost - (40.0 * 3e-6 + 60.0 * 4e-6) * 1.2).abs() < 1e-12);
    assert!((breakdown.reasoning_cost - 20.0 * 8e-6 * 1.2).abs() < 1e-12);
}

#[rstest]
fn token_type_cost_breakdown_is_provider_agnostic_for_perplexity() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "perplexity/sonar-reasoning".to_string(),
        json!({
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "output_cost_per_reasoning_token": 4e-6,
            "cache_read_input_token_cost": 1e-7,
            "litellm_provider": "perplexity"
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 2000,
        "total_tokens": 3000,
        "completion_tokens_details": {"reasoning_tokens": 400, "text_tokens": 1600},
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 1000}
    }}))
    .unwrap()
    .unwrap();
    let breakdown = get_token_type_cost_breakdown(
        &catalog,
        ModelCostRequest {
            model: "perplexity/sonar-reasoning",
            provider: Some("perplexity"),
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            region: None,
            at: at("2026-01-01T12:00Z"),
            response_time_ms: None,
        },
        None,
    );
    assert!((breakdown.reasoning_cost - 400.0 * 4e-6).abs() < 1e-12);
    assert!((breakdown.cache_read_cost - 0.0).abs() < 1e-12);
}

fn breakdown_rates() -> Value {
    json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 1e-7,
        "cache_creation_input_token_cost": 1.25e-6,
        "output_cost_per_reasoning_token": 3e-6
    })
}

fn usage_with_zero_details(private: (&str, Value)) -> ChatUsage {
    ChatUsage {
        prompt_tokens: 100,
        completion_tokens: 100,
        total_tokens: 200,
        prompt_tokens_details: Some(PromptTokenDetails::default()),
        completion_tokens_details: Some(CompletionTokenDetails {
            reasoning_tokens: Some(0),
            ..CompletionTokenDetails::default()
        }),
        extra: [(private.0.to_owned(), private.1)].into_iter().collect(),
        ..ChatUsage::default()
    }
}

#[rstest]
#[case::cache_read("_cache_read_input_tokens", json!(40), (0.0, 40.0 * 1e-7, 0.0))]
#[case::cache_creation("_cache_creation_input_tokens", json!(40), (0.0, 0.0, 40.0 * 1.25e-6))]
#[case::reasoning("reasoning_tokens", json!(40), (40.0 * 3e-6, 0.0, 0.0))]
#[case::bool_counts_as_one("reasoning_tokens", json!(true), (3e-6, 0.0, 0.0))]
#[case::float_is_not_an_int("reasoning_tokens", json!(40.0), (0.0, 0.0, 0.0))]
#[case::negative_counts_nothing("_cache_read_input_tokens", json!(-5), (0.0, 0.0, 0.0))]
fn calculate_token_type_cost_breakdown_falls_back_to_private_counters_when_details_are_zero(
    #[case] counter: &str,
    #[case] value: Value,
    #[case] expected: (f64, f64, f64),
) {
    let model_info = breakdown_rates();
    let usage = usage_with_zero_details((counter, value));
    let breakdown = calculate_token_type_cost_breakdown(BilledRatesRequest {
        model_info: Some(&model_info),
        usage: &usage,
        provider: Some("openai"),
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: at("2026-01-01T12:00Z"),
        custom_cost_per_token: None,
    });
    assert!((breakdown.reasoning_cost - expected.0).abs() < 1e-15);
    assert!((breakdown.cache_read_cost - expected.1).abs() < 1e-15);
    assert!((breakdown.cache_creation_cost - expected.2).abs() < 1e-15);
}
