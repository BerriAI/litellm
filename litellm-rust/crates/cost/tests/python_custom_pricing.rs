#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py::test_custom_pricing_anthropic_style_cache_tokens_not_double_counted
use litellm_cost::cost_calculator::{
    cost_per_token_for_call_with_custom, cost_per_token_with_custom,
};
use litellm_cost::error::CostError;

use std::collections::HashMap;

use litellm_cost::catalog::{CostCall, ModelCostRequest, ModelInfoCatalog};
use litellm_cost::custom_pricing::{
    CustomPricing, CustomTokenRates, RawUsage, cost_per_token_custom_pricing_helper,
    normalize_cache_usage,
};
use litellm_cost::usage_dispatch::chat_usage;
use rstest::rstest;
use serde_json::json;

fn usage(prompt_tokens: f64, completion_tokens: f64) -> RawUsage {
    RawUsage {
        prompt_tokens,
        completion_tokens,
        details_cached_tokens: None,
        details_cache_write_tokens: None,
        details_cache_creation_tokens: None,
        top_level_cache_read_tokens: None,
        top_level_cache_creation_tokens: None,
        fallback_cache_read_tokens: None,
        fallback_cache_creation_tokens: None,
    }
}

fn rates(
    input: f64,
    output: f64,
    cache_read: Option<f64>,
    cache_creation: Option<f64>,
) -> CustomTokenRates {
    CustomTokenRates {
        input,
        output,
        cache_read,
        cache_creation,
    }
}

#[rstest]
#[case::cache_read(
    RawUsage { details_cached_tokens: Some(3456.0), ..usage(6074.0, 285.0) },
    rates(2.5e-6, 15e-6, Some(0.25e-6), None),
    2618.0 * 2.5e-6 + 3456.0 * 0.25e-6,
    285.0 * 15e-6
)]
#[case::cache_creation(
    RawUsage {
        details_cached_tokens: Some(1000.0),
        details_cache_creation_tokens: Some(500.0),
        ..usage(4000.0, 100.0)
    },
    rates(2.5e-6, 15e-6, Some(0.25e-6), Some(3.125e-6)),
    2500.0 * 2.5e-6 + 1000.0 * 0.25e-6 + 500.0 * 3.125e-6,
    100.0 * 15e-6
)]
#[case::cache_write_alias(
    RawUsage {
        details_cached_tokens: Some(1000.0),
        details_cache_write_tokens: Some(500.0),
        ..usage(4000.0, 100.0)
    },
    rates(2.5e-6, 15e-6, Some(0.25e-6), Some(3.125e-6)),
    2500.0 * 2.5e-6 + 1000.0 * 0.25e-6 + 500.0 * 3.125e-6,
    100.0 * 15e-6
)]
#[case::anthropic_style(
    RawUsage {
        top_level_cache_read_tokens: Some(1500.0),
        top_level_cache_creation_tokens: Some(300.0),
        ..usage(2000.0, 100.0)
    },
    rates(3e-6, 15e-6, Some(0.3e-6), Some(3.75e-6)),
    2000.0 * 3e-6 + 1500.0 * 0.3e-6 + 300.0 * 3.75e-6,
    100.0 * 15e-6
)]
#[case::missing_cache_rates(
    RawUsage { details_cached_tokens: Some(400.0), ..usage(1000.0, 100.0) },
    rates(2.5e-6, 15e-6, None, None),
    1000.0 * 2.5e-6,
    100.0 * 15e-6
)]
fn cost_per_token_custom_pricing_helper_matches_python_cache_cases(
    #[case] raw: RawUsage,
    #[case] token_rates: CustomTokenRates,
    #[case] expected_input: f64,
    #[case] expected_output: f64,
) {
    let normalized = normalize_cache_usage(raw).unwrap();
    let cost = cost_per_token_custom_pricing_helper(
        normalized,
        CustomPricing {
            token: Some(token_rates),
            per_second: None,
        },
        None,
    )
    .unwrap()
    .unwrap();
    assert!((cost.input - expected_input).abs() < 1e-12);
    assert!((cost.output - expected_output).abs() < 1e-12);
}

#[rstest]
fn normalize_cache_usage_top_level_zero_overrides_details() {
    let raw = RawUsage {
        details_cached_tokens: Some(400.0),
        top_level_cache_read_tokens: Some(0.0),
        ..usage(1000.0, 0.0)
    };
    let normalized = normalize_cache_usage(raw).unwrap();
    assert_eq!(normalized.cached_tokens, 0.0);
    assert_eq!(normalized.prompt_tokens, 1000.0);
}

#[rstest]
fn normalize_cache_usage_fallback_cache_counts_use_anthropic_convention() {
    let raw = RawUsage {
        fallback_cache_read_tokens: Some(200.0),
        fallback_cache_creation_tokens: Some(50.0),
        ..usage(1000.0, 0.0)
    };
    let normalized = normalize_cache_usage(raw).unwrap();
    assert_eq!(normalized.prompt_tokens, 1250.0);
    assert_eq!(normalized.cached_tokens, 200.0);
    assert_eq!(normalized.cache_creation_tokens, 50.0);
}

#[rstest]
#[case(Some(2500.0), 0.2)]
#[case(None, 0.0)]
fn cost_per_token_custom_pricing_helper_prices_seconds(
    #[case] duration_ms: Option<f64>,
    #[case] expected_output: f64,
) {
    let normalized = normalize_cache_usage(usage(0.0, 0.0)).unwrap();
    let cost = cost_per_token_custom_pricing_helper(
        normalized,
        CustomPricing {
            token: None,
            per_second: Some(0.08),
        },
        duration_ms,
    )
    .unwrap()
    .unwrap();
    assert_eq!(cost.input, 0.0);
    assert!((cost.output - expected_output).abs() < 1e-12);
}

#[rstest]
fn cost_per_token_custom_pricing_helper_returns_none_without_custom_rates() {
    let normalized = normalize_cache_usage(usage(1000.0, 100.0)).unwrap();
    assert_eq!(
        cost_per_token_custom_pricing_helper(normalized, CustomPricing::NONE, None),
        Ok(None)
    );
}

#[rstest]
fn cost_per_token_custom_pricing_helper_prefers_token_rates_over_seconds() {
    let normalized = normalize_cache_usage(usage(1000.0, 100.0)).unwrap();
    let cost = cost_per_token_custom_pricing_helper(
        normalized,
        CustomPricing {
            token: Some(rates(1e-6, 2e-6, None, None)),
            per_second: Some(10.0),
        },
        Some(10_000.0),
    )
    .unwrap()
    .unwrap();
    assert!((cost.input - 0.001).abs() < 1e-12);
    assert!((cost.output - 0.0002).abs() < 1e-12);
}

#[rstest]
fn catalog_custom_rates_override_lookup_and_call_dispatch() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let usage = chat_usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
        "cost": 99.0
    }))
    .unwrap();
    let request = ModelCostRequest {
        model: "unregistered",
        provider: Some("openai"),
        region: None,
        usage: &usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse().unwrap(),
        response_time_ms: Some(2500.0),
    };
    let pricing = CustomPricing {
        token: Some(rates(0.01, 0.02, Some(0.001), Some(0.03))),
        per_second: Some(1.0),
    };
    let direct = cost_per_token_with_custom(&catalog, request, pricing).unwrap();
    let dispatched = cost_per_token_for_call_with_custom(
        &catalog,
        request,
        CostCall::Speech {
            prompt_characters: None,
        },
        pricing,
    )
    .unwrap();
    assert!((direct.0 - 0.104).abs() < 1e-12);
    assert!((direct.1 - 0.06).abs() < 1e-12);
    assert_eq!(direct, dispatched);
    let seconds = cost_per_token_with_custom(
        &catalog,
        request,
        CustomPricing {
            token: None,
            per_second: Some(0.08),
        },
    )
    .unwrap();
    assert_eq!(seconds.0, 0.0);
    assert!((seconds.1 - 0.2).abs() < 1e-12);
}

#[rstest]
fn catalog_custom_pricing_falls_back_and_rejects_invalid_rates() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/mapped".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
    )]));
    let usage = chat_usage(&json!({"prompt_tokens": 10, "completion_tokens": 3})).unwrap();
    let request = ModelCostRequest {
        model: "mapped",
        provider: Some("openai"),
        region: None,
        usage: &usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse().unwrap(),
        response_time_ms: None,
    };
    let fallback = cost_per_token_with_custom(&catalog, request, CustomPricing::NONE).unwrap();
    assert!((fallback.0 - 0.1).abs() < 1e-12);
    assert!((fallback.1 - 0.06).abs() < 1e-12);
    assert_eq!(
        cost_per_token_with_custom(
            &catalog,
            request,
            CustomPricing {
                token: Some(rates(-0.01, 0.02, None, None)),
                per_second: None,
            },
        ),
        Err(CostError::InvalidRate)
    );
}

#[rstest]
#[case::negative_prompt_tokens(usage(-1.0, 0.0), None)]
#[case::nan_completion_tokens(usage(0.0, f64::NAN), None)]
#[case::negative_detail_cache_tokens(RawUsage { details_cached_tokens: Some(-1.0), ..usage(10.0, 0.0) }, None)]
#[case::infinite_fallback_tokens(RawUsage { fallback_cache_read_tokens: Some(f64::INFINITY), ..usage(10.0, 0.0) }, None)]
#[case::overflowing_prompt_sum(RawUsage { top_level_cache_read_tokens: Some(f64::MAX), top_level_cache_creation_tokens: Some(f64::MAX), ..usage(f64::MAX, 0.0) }, None)]
#[case::valid_usage(usage(10.0, 2.0), Some((10.0, 2.0)))]
fn normalize_cache_usage_rejects_negative_or_non_finite_counts(
    #[case] raw: RawUsage,
    #[case] expected: Option<(f64, f64)>,
) {
    let normalized = normalize_cache_usage(raw);
    match expected {
        Some((prompt, completion)) => {
            let normalized = normalized.unwrap();
            assert_eq!(
                (normalized.prompt_tokens, normalized.completion_tokens),
                (prompt, completion)
            );
        }
        None => assert_eq!(normalized, Err(CostError::InvalidUsage)),
    }
}

#[rstest]
#[case::negative_token_rate(CustomPricing { token: Some(rates(0.1, -0.1, None, None)), per_second: None }, None, CostError::InvalidRate)]
#[case::nan_cache_read_rate(CustomPricing { token: Some(rates(0.1, 0.1, Some(f64::NAN), None)), per_second: None }, None, CostError::InvalidRate)]
#[case::negative_second_rate(CustomPricing { token: None, per_second: Some(-0.5) }, Some(10.0), CostError::InvalidRate)]
#[case::negative_duration(CustomPricing { token: None, per_second: Some(0.5) }, Some(-10.0), CostError::InvalidDuration)]
#[case::infinite_duration(CustomPricing { token: None, per_second: Some(0.5) }, Some(f64::INFINITY), CostError::InvalidDuration)]
#[case::overflowing_token_cost(CustomPricing { token: Some(rates(f64::MAX, 0.0, None, None)), per_second: None }, None, CostError::NonFiniteCost)]
fn cost_per_token_custom_pricing_helper_rejects_unusable_rates_and_durations(
    #[case] pricing: CustomPricing,
    #[case] response_time_ms: Option<f64>,
    #[case] expected: CostError,
) {
    let normalized = normalize_cache_usage(usage(1000.0, 0.0)).unwrap();
    assert_eq!(
        cost_per_token_custom_pricing_helper(normalized, pricing, response_time_ms),
        Err(expected)
    );
}
