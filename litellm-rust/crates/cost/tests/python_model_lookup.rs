#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_cost_per_token_duplicate_openai_prefix_matches_model_cost

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{CatalogError, CostCatalog, ModelCostRequest, ModelInfoCatalog};
use litellm_cost::usage_dispatch::get_usage_object;
use litellm_cost::{PromptConvention, Rate, Rates, Request, ServiceTier, ThresholdPolicy, Usage};
use rstest::rstest;
use serde_json::json;

fn rates(input: f64, output: f64) -> Rates {
    Rates {
        input: Rate::Value(input),
        output: Rate::Value(output),
        ..Rates::EMPTY
    }
}

fn catalog() -> CostCatalog {
    CostCatalog::new(HashMap::from([
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

fn request() -> Request {
    Request {
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cache_write_5m_tokens: None,
            cache_write_1h_tokens: None,
            prompt_convention: PromptConvention::IncludesCache,
        },
        service_tier: ServiceTier::Standard,
        threshold_policy: ThresholdPolicy::Exclusive,
        region_multiplier: None,
        billed_at_utc_minute: None,
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
        catalog().select_model_key(model, provider, region),
        expected
    );
}

#[rstest]
fn cost_per_token_uses_the_selected_regional_prices() {
    let cost = catalog()
        .cost_per_token(
            "bedrock_mantle/model",
            Some("bedrock_mantle"),
            Some("us-gov-west-1"),
            &request(),
        )
        .unwrap();
    assert!((cost.input() - 100.0 * 5e-6).abs() < 1e-12);
    assert!((cost.output() - 50.0 * 6e-6).abs() < 1e-12);
}

#[rstest]
fn cost_per_token_reports_an_unmapped_model() {
    assert_eq!(
        catalog().cost_per_token("missing", Some("openai"), None, &request()),
        Err(CatalogError::ModelNotFound)
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
    let actual = catalog
        .cost_per_token(ModelCostRequest {
            model: "openai/openai/model",
            provider: Some("openai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: Some("eu"),
            vertex_location: None,
            at,
            response_time_ms: None,
        })
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
    let actual = catalog
        .cost_per_token(ModelCostRequest {
            model: "model",
            provider: Some("xai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: None,
        })
        .unwrap();
    assert!((actual.0 - 128_000.0 * 5e-6).abs() < 1e-12);
    assert!((actual.1 - 10.0 * 7e-6).abs() < 1e-12);
}
