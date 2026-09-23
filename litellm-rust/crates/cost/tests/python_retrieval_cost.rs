use litellm_cost::retrieval_cost::{rerank_cost, vector_store_search_cost};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case("vertex_ai", json!({"input_cost_per_query": 0.25}), json!({"search_units": 3}), (0.75, 0.0))]
#[case("cohere", json!({"input_cost_per_query": 0.25}), json!({"search_units": 3}), (0.75, 0.0))]
#[case("jina_ai", json!({"input_cost_per_token": 0.01}), json!({"total_tokens": 30}), (0.3, 0.0))]
#[case("voyage", json!({"input_cost_per_token": 0.01}), json!({"total_tokens": 30}), (0.3, 0.0))]
#[case("jina_ai", json!({"input_cost_per_query": 0.25}), json!({"search_units": 3}), (0.0, 0.0))]
#[case("cohere", json!({"input_cost_per_token": 0.01}), json!({"total_tokens": 30}), (0.0, 0.0))]
fn rerank_cost_uses_provider_billing_units(
    #[case] provider: &str,
    #[case] model_info: serde_json::Value,
    #[case] billed_units: serde_json::Value,
    #[case] expected: (f64, f64),
) {
    let actual = rerank_cost(provider, Some(&model_info), Some(&billed_units));
    assert!((actual.0 - expected.0).abs() < 1e-12);
    assert_eq!(actual.1, expected.1);
}

#[rstest]
fn rerank_cost_returns_zero_when_pricing_or_units_are_missing() {
    assert_eq!(
        rerank_cost("cohere", None, Some(&json!({"search_units": 2}))),
        (0.0, 0.0)
    );
    assert_eq!(
        rerank_cost("cohere", Some(&json!({"input_cost_per_query": 0.25})), None),
        (0.0, 0.0)
    );
    assert_eq!(
        rerank_cost(
            "cohere",
            Some(&json!({"input_cost_per_query": 0.25})),
            Some(&json!({}))
        ),
        (0.0, 0.0)
    );
}

#[rstest]
#[case("vertex_ai", Some("search_api"), 0.25)]
#[case("vertex_ai", Some("rag_api"), 0.0)]
#[case("vertex_ai", None, 0.0)]
#[case("openai", Some("search_api"), 0.0)]
fn vector_store_search_cost_only_prices_vertex_search_api(
    #[case] provider: &str,
    #[case] api_type: Option<&str>,
    #[case] expected: f64,
) {
    assert_eq!(
        vector_store_search_cost(
            provider,
            api_type,
            Some(&json!({"input_cost_per_query": 0.25}))
        ),
        (expected, 0.0)
    );
}

#[rstest]
fn model_info_catalog_routes_retrieval_costs_from_selected_metadata() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "vertex_ai/model".to_owned(),
            json!({"input_cost_per_query": 0.25}),
        ),
        (
            "jina_ai/model".to_owned(),
            json!({"input_cost_per_token": 0.01}),
        ),
        (
            "vertex_ai/search_api".to_owned(),
            json!({"input_cost_per_query": 0.4}),
        ),
    ]));
    assert_eq!(
        catalog.rerank_cost(
            "model",
            "vertex_ai",
            None,
            Some(&json!({"search_units": 3}))
        ),
        (0.75, 0.0)
    );
    assert_eq!(
        catalog.rerank_cost("model", "jina_ai", None, Some(&json!({"total_tokens": 30}))),
        (0.3, 0.0)
    );
    assert_eq!(
        catalog.vector_store_search_cost("vertex_ai", Some("search_api")),
        (0.4, 0.0)
    );
    assert_eq!(
        catalog.rerank_cost(
            "missing",
            "vertex_ai",
            None,
            Some(&json!({"search_units": 3}))
        ),
        (0.0, 0.0)
    );
}
use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
