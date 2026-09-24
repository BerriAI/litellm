#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_route_image_generation_cost_honors_deployment_model_info

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::call_type::CallTypes;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::image_cost_router::{
    ImageCostRouteRequest, call_type_has_image_response, deployment_pricing,
    route_image_generation_cost_calculator,
};
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-09-22T12:00:00Z".parse().unwrap()
}

fn request<'a>(
    model: &'a str,
    provider: &'a str,
    response: &'a Value,
) -> ImageCostRouteRequest<'a> {
    ImageCostRouteRequest {
        model,
        provider: Some(provider),
        image_response: response,
        call_type: Some(CallTypes::image_generation),
        quality: None,
        size: None,
        n: None,
        optional_params: &Value::Null,
        supplied_model_info: None,
        at: at(),
    }
}

#[rstest]
#[case("image_generation", true)]
#[case("aimage_generation", true)]
#[case("passthrough-image-generation", true)]
#[case("image_edit", true)]
#[case("aimage_edit", true)]
#[case("completion", false)]
fn call_type_has_image_response_matches_python_call_types(
    #[case] call_type: &str,
    #[case] expected: bool,
) {
    assert_eq!(call_type_has_image_response(call_type), expected);
}

#[rstest]
#[case("recraft")]
#[case("aiml")]
#[case("cometapi")]
#[case("runwayml")]
fn route_image_generation_prices_flat_providers_with_deployment_rate(#[case] provider: &str) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("{provider}/model"),
        json!({"output_cost_per_image": 0.25}),
    )]));
    let response = json!({"data": [{}, {}]});
    let supplied = json!({"output_cost_per_image": 0.4});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            supplied_model_info: Some(&supplied),
            ..request("model", provider, &response)
        },
    )
    .unwrap();
    assert_eq!(cost, 0.8);
}

#[rstest]
fn deployment_pricing_ignores_invalid_override_and_normalizes_numeric_string() {
    let supplied = json!({
        "output_cost_per_image": "not-a-price",
        "input_cost_per_pixel": "0.0002",
        "web_search_billing_unit": "per_query"
    });
    assert_eq!(
        deployment_pricing(Some(&supplied)),
        Some(json!({"input_cost_per_pixel": 0.0002}))
    );
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "recraft/model".to_owned(),
        json!({"output_cost_per_image": 0.25}),
    )]));
    let response = json!({"data": [{}, {}]});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            supplied_model_info: Some(&supplied),
            ..request("model", "recraft", &response)
        },
    )
    .unwrap();
    assert_eq!(cost, 0.5);
}

#[rstest]
#[case("gemini", "image_generation")]
#[case("gemini", "image_edit")]
#[case("vertex_ai", "image_generation")]
fn route_google_image_calls_keep_token_and_search_pricing(
    #[case] provider: &str,
    #[case] call_type: &str,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("{provider}/model"),
        json!({
            "input_cost_per_token": 0.001,
            "output_cost_per_image_token": 0.003,
            "output_cost_per_image": 0.5,
            "web_search_billing_unit": "per_query",
            "search_context_cost_per_query": {"search_context_size_medium": 0.02}
        }),
    )]));
    let response = json!({"data": [{}], "usage": {
        "input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "web_search_requests": 2
    }});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            call_type: call_type.parse::<CallTypes>().ok(),
            ..request("model", provider, &response)
        },
    )
    .unwrap();
    assert!((cost - (3.0 * 0.001 + 2.0 * 0.003 + 2.0 * 0.02)).abs() < 1e-12);
}

#[rstest]
fn route_azure_ai_image_calls_use_requested_dimensions() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "azure_ai/model".to_owned(),
        json!({"input_cost_per_pixel": 0.0001}),
    )]));
    let response = json!({"data": [{}, {}], "size": "3x4"});
    let params = json!({"width": 10, "height": 20});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            optional_params: &params,
            ..request("model", "azure_ai", &response)
        },
    )
    .unwrap();
    assert!((cost - 2.0 * 10.0 * 20.0 * 0.0001).abs() < 1e-12);
}

#[rstest]
fn route_azure_ai_image_uses_optional_size_when_response_has_none() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "azure_ai/model".to_owned(),
        json!({"input_cost_per_pixel": 0.0001}),
    )]));
    let response = json!({"data": [{}, {}]});
    let params = json!({"size": "20x10"});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            optional_params: &params,
            ..request("model", "azure_ai", &response)
        },
    )
    .unwrap();
    assert!((cost - 2.0 * 20.0 * 10.0 * 0.0001).abs() < 1e-12);
}

#[rstest]
#[case(None, Some("low"), 0.04)]
#[case(Some("high"), Some("low"), 0.08)]
#[case(None, None, 0.06)]
fn route_default_image_cost_uses_response_then_requested_quality(
    #[case] response_quality: Option<&str>,
    #[case] requested_quality: Option<&str>,
    #[case] expected: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "xai/model".to_owned(),
            json!({"input_cost_per_image": 0.06}),
        ),
        (
            "low/1024-x-1024/model".to_owned(),
            json!({"input_cost_per_image": 0.04}),
        ),
        (
            "high/1024-x-1024/model".to_owned(),
            json!({"input_cost_per_image": 0.08}),
        ),
    ]));
    let response = json!({"data": [{}], "quality": response_quality});
    let params = json!({"quality": requested_quality});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            optional_params: &params,
            ..request("xai/model", "xai", &response)
        },
    )
    .unwrap();
    assert_eq!(cost, expected);
}

#[rstest]
fn route_default_image_cost_uses_requested_size_and_deployment_override() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "low/1536-x-1024/model".to_owned(),
        json!({"input_cost_per_image": 0.05}),
    )]));
    let response = json!({"data": [{}, {}]});
    let params = json!({"quality": "low", "size": "1536x1024"});
    let supplied = json!({"input_cost_per_image": "0.07"});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            optional_params: &params,
            supplied_model_info: Some(&supplied),
            ..request("xai/model", "xai", &response)
        },
    )
    .unwrap();
    assert_eq!(cost, 0.14);
}

#[rstest]
fn deployment_pricing_reads_only_the_python_pricing_field_set() {
    let model_info = json!({
        "output_cost_per_image": 0.1,
        "cache_read_input_token_cost": 1e-7,
        "future_vendor_cost_per_widget": 5.0
    });
    let pricing = deployment_pricing(Some(&model_info)).expect("pricing");
    assert_eq!(pricing.get("output_cost_per_image"), Some(&json!(0.1)));
    assert_eq!(
        pricing.get("cache_read_input_token_cost"),
        Some(&json!(1e-7))
    );
    assert_eq!(pricing.get("future_vendor_cost_per_widget"), None);
}

fn per_size_catalog() -> ModelInfoCatalog {
    ModelInfoCatalog::new(HashMap::from([
        (
            "standard/1024-x-1024/dall-e".to_owned(),
            json!({"input_cost_per_image": 0.04}),
        ),
        (
            "standard/512-x-512/dall-e".to_owned(),
            json!({"input_cost_per_image": 0.02}),
        ),
        (
            "hd/1024-x-1024/dall-e".to_owned(),
            json!({"input_cost_per_image": 0.08}),
        ),
    ]))
}

#[rstest]
#[case::count_from_response_data(json!({"data": [{}, {}]}), None, 0.08)]
#[case::explicit_count_wins(json!({"data": [{}, {}]}), Some(3), 0.12)]
#[case::no_data_bills_no_images(json!({}), None, 0.0)]
#[case::empty_data_bills_no_images(json!({"data": []}), None, 0.0)]
fn route_image_generation_counts_images_like_python(
    #[case] response: Value,
    #[case] n: Option<u64>,
    #[case] expected: f64,
) {
    let cost = route_image_generation_cost_calculator(
        &per_size_catalog(),
        ImageCostRouteRequest {
            n,
            ..request("dall-e", "custom", &response)
        },
    )
    .unwrap();
    assert!((cost - expected).abs() < 1e-12);
}

#[rstest]
#[case::empty_size_falls_through_to_response(Some(""), None, json!({"data": [{}], "size": "512x512"}), json!({}), 0.02)]
#[case::empty_quality_falls_through_to_params(None, Some(""), json!({"data": [{}]}), json!({"quality": "hd"}), 0.08)]
fn route_image_generation_skips_empty_strings_like_python_or(
    #[case] size: Option<&str>,
    #[case] quality: Option<&str>,
    #[case] response: Value,
    #[case] params: Value,
    #[case] expected: f64,
) {
    let cost = route_image_generation_cost_calculator(
        &per_size_catalog(),
        ImageCostRouteRequest {
            size,
            quality,
            optional_params: &params,
            ..request("dall-e", "custom", &response)
        },
    )
    .unwrap();
    assert!((cost - expected).abs() < 1e-12);
}
