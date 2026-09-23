use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::azure_cost::output_per_second_cost;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::databricks_cost::registry_key;
use litellm_cost::responses_usage::ChatUsage;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn request<'a>(model: &'a str, provider: &'a str, usage: &'a ChatUsage) -> ModelCostRequest<'a> {
    ModelCostRequest {
        model,
        provider: Some(provider),
        region: None,
        usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-09-22T12:00:00Z".parse::<Timestamp>().unwrap(),
        response_time_ms: None,
    }
}

#[rstest]
#[case(json!({"output_cost_per_second": 0.02}), Some(1500.0), Some((0.0, 0.03)))]
#[case(json!({"output_cost_per_second": 0.0}), Some(1500.0), Some((0.0, 0.0)))]
#[case(json!({"output_cost_per_second": 0.02}), None, None)]
#[case(json!({"output_cost_per_second": null}), Some(1500.0), None)]
fn azure_output_per_second_cost_requires_rate_and_response_time(
    #[case] model_info: Value,
    #[case] response_time_ms: Option<f64>,
    #[case] expected: Option<(f64, f64)>,
) {
    let actual = output_per_second_cost(&model_info, response_time_ms);
    assert_eq!(actual, expected);
}

#[rstest]
fn azure_provider_output_seconds_override_token_rates_when_present() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "azure/speech".to_owned(),
        json!({"mode": "audio_speech", "input_cost_per_token": 0.001, "output_cost_per_token": 0.002, "output_cost_per_second": 0.02}),
    )]));
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
            .unwrap()
            .unwrap();
    let seconds = catalog
        .cost_per_token(ModelCostRequest {
            response_time_ms: Some(1500.0),
            ..request("speech", "azure", &usage)
        })
        .unwrap();
    let tokens = catalog
        .cost_per_token(request("speech", "azure", &usage))
        .unwrap();
    assert_eq!(seconds, (0.0, 0.03));
    assert_eq!(tokens, (0.1, 0.04));
}

#[rstest]
#[case("databricks/dbrx-instruct-fast", "databricks-dbrx-instruct")]
#[case(
    "meta-llama-3.1-70b-instruct-v2",
    "databricks-meta-llama-3-1-70b-instruct"
)]
#[case("databricks/custom-endpoint", "custom-endpoint")]
fn databricks_registry_key_maps_legacy_prefixes(#[case] model: &str, #[case] expected: &str) {
    assert_eq!(registry_key(model), expected);
}

#[rstest]
fn databricks_provider_uses_legacy_registry_price_and_preserves_original_duration_override() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "databricks/databricks-dbrx-instruct".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
        (
            "databricks/dbrx-instruct-fast".to_owned(),
            json!({"mode": "chat", "input_cost_per_second": 0.05}),
        ),
    ]));
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
            .unwrap()
            .unwrap();
    let alias = catalog
        .cost_per_token(request("dbrx-instruct-slow", "databricks", &usage))
        .unwrap();
    let duration = catalog
        .cost_per_token(ModelCostRequest {
            response_time_ms: Some(2000.0),
            ..request("dbrx-instruct-fast", "databricks", &usage)
        })
        .unwrap();
    assert_eq!(alias, (0.2, 0.06));
    assert_eq!(duration, (0.1, 0.0));
}

#[rstest]
fn lemonade_provider_is_free_for_unmapped_and_token_priced_models() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "lemonade/priced".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
        (
            "lemonade/duration".to_owned(),
            json!({"mode": "chat", "input_cost_per_second": 0.05}),
        ),
    ]));
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
            .unwrap()
            .unwrap();
    assert_eq!(
        catalog
            .cost_per_token(request("unknown", "lemonade", &usage))
            .unwrap(),
        (0.0, 0.0)
    );
    assert_eq!(
        catalog
            .cost_per_token(request("priced", "lemonade", &usage))
            .unwrap(),
        (0.0, 0.0)
    );
    assert_eq!(
        catalog.lemonade_cost_per_token(request("duration", "lemonade", &usage)),
        (0.0, 0.0)
    );
    let elapsed = catalog
        .cost_per_token(ModelCostRequest {
            response_time_ms: Some(2000.0),
            ..request("duration", "lemonade", &usage)
        })
        .unwrap();
    assert_eq!(elapsed, (0.1, 0.0));
}
