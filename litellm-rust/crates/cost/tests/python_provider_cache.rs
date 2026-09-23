use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::provider_cache::with_default_cache_read_rate;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

#[rstest]
fn with_default_cache_read_rate_derives_base_and_off_peak_rates() {
    let info = json!({
        "input_cost_per_token": 2e-6,
        "off_peak_pricing": {"hours_utc": "16:00-20:00", "input_cost_per_token": 1e-6}
    });
    let actual = with_default_cache_read_rate(&info);
    assert_eq!(actual["cache_read_input_token_cost"], json!(1e-6));
    assert_eq!(
        actual["off_peak_pricing"]["cache_read_input_token_cost"],
        json!(0.5e-6)
    );
    assert!(info.get("cache_read_input_token_cost").is_none());
}

#[rstest]
fn with_default_cache_read_rate_preserves_explicit_cache_rates() {
    let info = json!({
        "input_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 0.25e-6,
        "off_peak_pricing": {"input_cost_per_token": 1e-6}
    });
    assert_eq!(with_default_cache_read_rate(&info), info);
}

#[rstest]
fn model_info_catalog_uses_fireworks_default_for_cached_tokens() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "fireworks_ai/model".to_owned(),
        json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6}),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 0,
        "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40}
    }}))
    .unwrap()
    .unwrap();
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let actual = catalog
        .cost_per_token(ModelCostRequest {
            model: "model",
            provider: Some("fireworks_ai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: None,
        })
        .unwrap();
    assert!((actual.0 - (60.0 * 2e-6 + 40.0 * 1e-6)).abs() < 1e-12);
}
