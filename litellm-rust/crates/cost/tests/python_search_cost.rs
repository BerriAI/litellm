#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/bedrock/search/test_agentcore_search_transformation.py::TestAgentCoreSearchEdgeCases::test_search_cost_lookup_is_mapped
// mirrors: unit/llms/parallel_ai/test_parallel_ai_search.py::test_explicit_mode_wins_over_processor

use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::search_cost::{
    ParallelAiPricing, effective_max_results, effective_mode, parallel_ai_search_cost,
    provider_usage, search_provider_cost_from_model_info, search_provider_cost_per_query,
    usage_count,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(10, 0.002)]
#[case(11, 0.004)]
#[case(50, 0.004)]
fn search_provider_cost_per_query_selects_inclusive_result_tier_and_last_fallback(
    #[case] max_results: u64,
    #[case] expected_rate: f64,
) {
    let model_info = json!({
        "input_cost_per_query": 0.001,
        "tiered_pricing": [
            {"max_results_range": [1, 10], "input_cost_per_query": 0.002},
            {"max_results_range": [11, 20], "input_cost_per_query": 0.004}
        ]
    });
    assert_eq!(
        search_provider_cost_from_model_info(&model_info, 3, &json!({"max_results": max_results})),
        (3.0 * expected_rate, 0.0)
    );
}

#[rstest]
fn search_provider_cost_per_query_uses_flat_rate_without_tiers() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "exa_ai/search".to_owned(),
        json!({"input_cost_per_query": 0.003}),
    )]));
    assert_eq!(
        search_provider_cost_per_query(&catalog, "search", Some("exa_ai"), 4, &json!({})).unwrap(),
        (0.012, 0.0)
    );
}

#[rstest]
#[case(json!({"mode": "fast"}), "fast")]
#[case(json!({"mode": "turbo", "processor": "pro"}), "turbo")]
#[case(json!({"processor": "pro"}), "advanced")]
#[case(json!({}), "basic")]
fn parallel_ai_effective_mode_preserves_explicit_mode_priority(
    #[case] params: Value,
    #[case] expected: &str,
) {
    assert_eq!(effective_mode(&params), expected);
}

#[rstest]
#[case(json!({"advanced_settings": {"max_results": 12}, "max_results": 18}), 12)]
#[case(json!({"advanced_settings": {"max_results": -1}, "max_results": 18}), 18)]
#[case(json!({"max_results": 0}), 0)]
#[case(json!({}), 10)]
fn parallel_ai_effective_max_results_uses_valid_nested_or_top_level_count(
    #[case] params: Value,
    #[case] expected: u64,
) {
    assert_eq!(effective_max_results(&params, 10), expected);
}

#[rstest]
fn parallel_ai_search_cost_prefers_billed_skus_over_requested_result_count() {
    let params = json!({
        "mode": "fast",
        "max_results": 50,
        "_parallel_ai_usage": [
            {"name": "sku_search", "count": 2},
            {"name": "sku_search", "count": 1},
            {"name": "sku_search_additional_results", "count": 4},
            {"name": "sku_search_additional_results", "count": 3},
            {"name": "sku_search_additional_results", "count": -2}
        ]
    });
    let usage = provider_usage(&params).unwrap();
    assert_eq!(usage_count(usage, "sku_search"), Some(3));
    assert_eq!(usage_count(usage, "sku_search_additional_results"), Some(7));
    let price = ParallelAiPricing {
        request_cost: 0.02,
        default_results: 10,
        additional_result_cost: 0.003,
    };
    assert!(
        (parallel_ai_search_cost(&params, Some(usage), price) - (3.0 * 0.02 + 7.0 * 0.003)).abs()
            < 1e-12
    );
    assert!((parallel_ai_search_cost(&params, None, price) - (0.02 + 40.0 * 0.003)).abs() < 1e-12);
}

#[rstest]
fn parallel_ai_catalog_selects_mode_price_and_zero_additional_results_for_usage_without_sku() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "parallel_ai/search".to_owned(),
            json!({"input_cost_per_query": 0.01}),
        ),
        (
            "parallel_ai/search-fast".to_owned(),
            json!({"input_cost_per_query": 0.02}),
        ),
        (
            "parallel_ai/search-turbo".to_owned(),
            json!({"input_cost_per_query": 0.03}),
        ),
    ]));
    let params = json!({
        "mode": "turbo",
        "max_results": 100,
        "_parallel_ai_usage": [{"name": "unrelated_sku", "count": 5}]
    });
    assert_eq!(
        search_provider_cost_per_query(&catalog, "ignored", Some("parallel_ai"), 9, &params)
            .unwrap(),
        (0.03, 0.0)
    );
}

#[rstest]
#[case(json!({"_parallel_ai_usage": [1, 2]}))]
#[case(json!({"_parallel_ai_usage": {"name": "sku_search", "count": 2}}))]
fn parallel_ai_invalid_usage_falls_back_to_requested_results(#[case] params: Value) {
    assert!(provider_usage(&params).is_none());
    let pricing = ParallelAiPricing {
        request_cost: 0.02,
        default_results: 10,
        additional_result_cost: 0.003,
    };
    assert_eq!(
        parallel_ai_search_cost(&params, provider_usage(&params), pricing),
        0.02
    );
}

#[rstest]
#[case::float_max_results(json!({"max_results": 50.0}), 0.025)]
#[case::float_tier_bounds_match_integer_request(json!({"max_results": 30}), 0.025)]
#[case::default_ten_results(json!({}), 0.005)]
#[case::boolean_is_one_result(json!({"max_results": true}), 0.005)]
fn search_provider_cost_compares_result_tiers_as_python_numbers(
    #[case] params: Value,
    #[case] expected_rate: f64,
) {
    let model_info = json!({
        "tiered_pricing": [
            {"max_results_range": [0, 25], "input_cost_per_query": 0.005},
            {"max_results_range": [26.0, 100.0], "input_cost_per_query": 0.025}
        ]
    });
    assert_eq!(
        search_provider_cost_from_model_info(&model_info, 1, &params),
        (expected_rate, 0.0)
    );
}
