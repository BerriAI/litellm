use std::collections::HashMap;

use jiff::Timestamp;
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
        call_type: Some("image_generation"),
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
            call_type: Some(call_type),
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
