#![allow(clippy::disallowed_types)]
// mirrors: local_testing/test_completion_cost.py::test_together_ai_qwen_completion_cost
use litellm_cost::cost_calculator::cost_per_token;
use litellm_cost::error::CostError;

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::together_cost::{
    TogetherThresholds, get_model_params_and_category, get_model_params_and_category_embeddings,
    has_together_registry_pricing, together_pricing_model,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn request<'a>(
    model: &'a str,
    usage: &'a litellm_cost::responses_usage::ChatUsage,
) -> ModelCostRequest<'a> {
    ModelCostRequest {
        model,
        provider: Some("together_ai"),
        region: None,
        usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse::<Timestamp>().unwrap(),
        response_time_ms: None,
    }
}

fn priced_as_completion_cost(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call_type: &str,
) -> Result<(f64, f64), CostError> {
    let pricing_model = together_pricing_model(catalog, request.model, request.provider, call_type);
    cost_per_token(
        catalog,
        ModelCostRequest {
            model: &pricing_model,
            ..request
        },
    )
}

fn usage() -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    }))
    .unwrap()
    .unwrap()
}

#[rstest]
#[case("MODEL-4B", "together-ai-up-to-4b")]
#[case("model-8b", "together-ai-4.1b-8b")]
#[case("model-21b", "together-ai-8.1b-21b")]
#[case("model-41b", "together-ai-21.1b-41b")]
#[case("model-80b", "together-ai-41.1b-80b")]
#[case("model-110b", "together-ai-81.1b-110b")]
#[case("model-111b", "model-111b")]
#[case("model-without-size", "model-without-size")]
#[case("model-7b-and-70b", "together-ai-4.1b-8b")]
fn chat_size_uses_first_parameter_match_and_python_thresholds(
    #[case] model: &str,
    #[case] expected: &str,
) {
    assert_eq!(
        get_model_params_and_category(model, "completion", TogetherThresholds::default()),
        expected
    );
}

#[rstest]
#[case("embedding", "MODEL-150M", "together-ai-embedding-up-to-150m")]
#[case("aembedding", "model-350m", "together-ai-embedding-151m-to-350m")]
#[case("embedding", "model-351m", "model-351m")]
fn embedding_size_uses_embedding_thresholds(
    #[case] call_type: &str,
    #[case] model: &str,
    #[case] expected: &str,
) {
    assert_eq!(
        get_model_params_and_category(model, call_type, TogetherThresholds::default()),
        expected
    );
    assert_eq!(
        get_model_params_and_category_embeddings(model, TogetherThresholds::default()),
        expected
    );
}

#[rstest]
fn configured_thresholds_change_category_selection() {
    let thresholds = TogetherThresholds {
        chat: [8, 16, 21, 41, 80, 110],
        embedding: [200, 400],
    };
    assert_eq!(
        get_model_params_and_category("model-8b", "completion", thresholds),
        "together-ai-up-to-4b"
    );
    assert_eq!(
        get_model_params_and_category("model-200m", "embedding", thresholds),
        "together-ai-embedding-up-to-150m"
    );
}

#[rstest]
#[case(json!({"input_cost_per_token": 0.0}), true)]
#[case(json!({"input_cost_per_token": null}), true)]
#[case(json!({"output_cost_per_token": 1e-6}), false)]
fn exact_registry_pricing_requires_input_rate_key(#[case] entry: Value, #[case] expected: bool) {
    let catalog = HashMap::from([("together_ai/model-7b".to_owned(), entry)]);
    assert_eq!(
        has_together_registry_pricing("model-7b", &catalog),
        expected
    );
    assert_eq!(
        has_together_registry_pricing("together_ai/model-7b", &catalog),
        expected
    );
}

#[rstest]
#[case(true, 5e-6)]
#[case(false, 2e-6)]
fn catalog_uses_exact_price_or_chat_category(
    #[case] exact_has_input_rate: bool,
    #[case] expected_input_rate: f64,
) {
    let exact = if exact_has_input_rate {
        json!({"input_cost_per_token": 5e-6, "output_cost_per_token": 6e-6})
    } else {
        json!({"mode": "chat"})
    };
    let catalog = ModelInfoCatalog::new(HashMap::from([
        ("together_ai/model-7b".to_owned(), exact),
        (
            "together-ai-4.1b-8b".to_owned(),
            json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 3e-6}),
        ),
    ]));
    let usage = usage();
    let (prompt, completion) =
        priced_as_completion_cost(&catalog, request("model-7b", &usage), "completion").unwrap();
    assert!((prompt - 100.0 * expected_input_rate).abs() < 1e-12);
    let output_rate = if exact_has_input_rate { 6e-6 } else { 3e-6 };
    assert!((completion - 20.0 * output_rate).abs() < 1e-12);
}

#[rstest]
fn catalog_routes_embedding_to_size_category() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "together-ai-embedding-151m-to-350m".to_owned(),
        json!({"input_cost_per_token": 4e-8, "output_cost_per_token": 0.0}),
    )]));
    let usage = usage();
    let (prompt, completion) =
        priced_as_completion_cost(&catalog, request("model-200m", &usage), "aembedding").unwrap();
    assert!((prompt - 100.0 * 4e-8).abs() < 1e-12);
    assert_eq!(completion, 0.0);
}

#[rstest]
fn catalog_does_not_price_metadata_only_exact_row_when_category_is_missing() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "together_ai/model-7b".to_owned(),
        json!({"mode": "chat"}),
    )]));
    let usage = usage();
    assert_eq!(
        priced_as_completion_cost(&catalog, request("model-7b", &usage), "completion"),
        Err(CostError::ModelNotFound)
    );
}

#[rstest]
fn an_embedding_without_a_megabyte_size_is_not_remapped_as_chat() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "together-ai-4.1b-8b".to_owned(),
        json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 3e-6}),
    )]));
    let usage = usage();
    assert_eq!(
        priced_as_completion_cost(&catalog, request("foo-7b-embed", &usage), "embedding"),
        Err(CostError::ModelNotFound)
    );
}

#[rstest]
fn per_second_pricing_is_read_from_the_size_category_not_the_unpriced_model() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "together_ai/foo-7b".to_owned(),
            json!({"input_cost_per_second": 0.01}),
        ),
        (
            "together-ai-4.1b-8b".to_owned(),
            json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 3e-6}),
        ),
    ]));
    let usage = usage();
    let (prompt, completion) = priced_as_completion_cost(
        &catalog,
        ModelCostRequest {
            response_time_ms: Some(1000.0),
            ..request("foo-7b", &usage)
        },
        "completion",
    )
    .unwrap();
    assert!((prompt - 100.0 * 2e-6).abs() < 1e-12);
    assert!((completion - 20.0 * 3e-6).abs() < 1e-12);
}

#[rstest]
#[case::registry_priced_model_keeps_its_name(Some("together_ai"), "model-7b", true, "model-7b")]
#[case::unpriced_together_model_maps_to_size(
    Some("together_ai"),
    "model-7b",
    false,
    "together-ai-4.1b-8b"
)]
#[case::together_name_without_provider(
    None,
    "togethercomputer/model-7b",
    false,
    "together-ai-4.1b-8b"
)]
#[case::other_provider_keeps_its_name(Some("openai"), "model-7b", false, "model-7b")]
fn together_pricing_model_maps_only_unpriced_together_models(
    #[case] provider: Option<&str>,
    #[case] model: &str,
    #[case] registry_priced: bool,
    #[case] expected: &str,
) {
    let catalog = ModelInfoCatalog::new(
        registry_priced
            .then(|| {
                (
                    format!("together_ai/{model}"),
                    json!({"input_cost_per_token": 1e-6}),
                )
            })
            .into_iter()
            .collect(),
    );
    assert_eq!(
        together_pricing_model(&catalog, model, provider, "completion"),
        expected
    );
}
