#![allow(clippy::disallowed_types)]

// mirrors: unit/llms/fireworks_ai/test_fireworks_ai_cost_calculator.py::test_fireworks_cache_read_rates_match_breakdown_and_caching_savings

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::generic_cost::calculate_generic_cost_from_model_info_with_region;
use litellm_cost::prompt_caching_savings::{
    PromptCachingSavingsRequest, calculate_prompt_caching_savings,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn usage(value: &Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

fn at(value: &str) -> Timestamp {
    value.parse().unwrap()
}

#[rstest]
#[case(20, 0, 0, 100, "2026-01-01T12:00Z", (None, None), None)]
#[case(0, 10, 15, 100, "2026-01-01T12:00Z", (None, None), None)]
#[case(20, 10, 15, 100, "2026-01-01T12:00Z", (Some("eu"), None), None)]
#[case(20, 10, 15, 100, "2026-01-01T18:00Z", (None, Some("us-east5")), None)]
#[case(20, 10, 15, 150, "2026-01-01T12:00Z", (None, None), None)]
#[case(20, 10, 15, 100, "2026-01-01T12:00Z", (None, None), Some("priority"))]
fn calculate_prompt_caching_savings_matches_uncached_bill_minus_cached_bill(
    #[case] reads: u64,
    #[case] writes_5m: u64,
    #[case] writes_1h: u64,
    #[case] prompt_tokens: u64,
    #[case] billed_at: &str,
    #[case] region: (Option<&str>, Option<&str>),
    #[case] service_tier: Option<&str>,
) {
    let (data_residency, vertex_location) = region;
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "cache_read_input_token_cost": 0.5e-6,
        "cache_creation_input_token_cost": 3e-6,
        "cache_creation_input_token_cost_above_1hr": 4e-6,
        "input_cost_per_token_priority": 3e-6,
        "cache_read_input_token_cost_priority": 0.75e-6,
        "cache_creation_input_token_cost_priority": 4.5e-6,
        "input_cost_per_token_above_120_tokens": 3e-6,
        "cache_read_input_token_cost_above_120_tokens": 0.75e-6,
        "cache_creation_input_token_cost_above_120_tokens": 4.5e-6,
        "cache_creation_input_token_cost_above_1hr_above_120_tokens": 6e-6,
        "regional_processing_uplift_multiplier_eu": 1.2,
        "regional_endpoint_uplift_multiplier": 1.3,
        "off_peak_pricing": {
            "hours_utc": "16:30-00:30",
            "input_cost_per_token": 1e-6,
            "cache_read_input_token_cost": 0.25e-6,
            "cache_creation_input_token_cost": 1.5e-6,
            "cache_creation_input_token_cost_above_1hr": 2e-6
        }
    });
    let cached = usage(&json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 8,
        "total_tokens": prompt_tokens + 8,
        "prompt_tokens_details": {
            "text_tokens": prompt_tokens - reads - writes_5m - writes_1h,
            "cached_tokens": reads,
            "cache_creation_tokens": writes_5m + writes_1h,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": writes_5m,
                "ephemeral_1h_input_tokens": writes_1h
            }
        }
    }));
    let uncached = usage(&json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 8,
        "total_tokens": prompt_tokens + 8,
        "prompt_tokens_details": {"text_tokens": prompt_tokens}
    }));
    let billed_at = at(billed_at);
    let charged = |usage| {
        let (input, output) = calculate_generic_cost_from_model_info_with_region(
            usage,
            &model_info,
            service_tier,
            false,
            data_residency,
            vertex_location,
            billed_at,
        );
        input + output
    };
    let expected = charged(&uncached) - charged(&cached);
    let savings = calculate_prompt_caching_savings(PromptCachingSavingsRequest {
        model_info: &model_info,
        usage: &cached,
        provider: Some("openai"),
        service_tier,
        data_residency,
        vertex_location,
        at: billed_at,
    });
    assert!((savings - expected).abs() < 1e-12);
    let catalog = ModelInfoCatalog::new(HashMap::from([("openai/model".to_owned(), model_info)]));
    assert_eq!(
        litellm_cost::prompt_caching_savings::prompt_caching_savings_for_model(
            &catalog,
            ModelCostRequest {
                model: "model",
                provider: Some("openai"),
                region: None,
                usage: &cached,
                service_tier,
                data_residency,
                vertex_location,
                at: billed_at,
                response_time_ms: None,
            }
        ),
        Some(savings)
    );
}

#[rstest]
#[case(0.0, 0.0, 40.0e-6)]
#[case(0.0, 3e-6, 20.0e-6)]
#[case(4e-6, 0.0, 0.0)]
fn calculate_prompt_caching_savings_respects_unpublished_and_expensive_cache_rates(
    #[case] read_rate: f64,
    #[case] write_rate: f64,
    #[case] expected: f64,
) {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "cache_read_input_token_cost": read_rate,
        "cache_creation_input_token_cost": write_rate
    });
    let usage = usage(&json!({
        "prompt_tokens": 100,
        "completion_tokens": 0,
        "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 20, "cache_creation_tokens": 20}
    }));
    let savings = calculate_prompt_caching_savings(PromptCachingSavingsRequest {
        model_info: &model_info,
        usage: &usage,
        provider: Some("openai"),
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: at("2026-01-01T12:00Z"),
    });
    assert!((savings - expected).abs() < 1e-12);
}
