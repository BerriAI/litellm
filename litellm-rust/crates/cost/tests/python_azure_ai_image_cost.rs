#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/azure_ai/image_generation/test_mai_image_generation.py::TestAzureMAIImageGeneration::test_mai_image_cost_calculator_token_based
// mirrors: test_litellm/llms/azure_ai/image_generation/test_azure_ai_flux2_image_generation.py::test_flux2_flex_cost_prefers_deployment_input_cost_per_pixel

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::azure_ai_image_cost::{
    AzureAiImageRequest, cost_calculator, input_cost_per_pixel,
};
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::image_cost_router::AzureAiImageCatalogRequest;
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-09-22T12:00:00Z".parse().unwrap()
}

fn catalog_with(model_info: &Value) -> ModelInfoCatalog {
    ModelInfoCatalog::new(HashMap::from([(
        "azure_ai/model".to_owned(),
        model_info.clone(),
    )]))
}

fn request<'a>(
    catalog: &'a ModelInfoCatalog,
    image_response: &'a Value,
    model_info: &'a Value,
) -> AzureAiImageRequest<'a> {
    AzureAiImageRequest {
        catalog,
        model: "azure_ai/model",
        image_response,
        model_info,
        supplied_model_info: None,
        shared_pricing: Some(model_info),
        size: None,
        n: None,
        optional_params: &Value::Null,
        at: at(),
    }
}

#[rstest]
fn azure_image_cost_prefers_token_usage_over_image_and_pixel_rates() {
    let info = json!({
        "input_cost_per_token": 0.001,
        "input_cost_per_image_token": 0.002,
        "output_cost_per_image_token": 0.003,
        "output_cost_per_image": 0.5,
        "input_cost_per_pixel": 0.0001
    });
    let response = json!({"data": [{}, {}], "usage": {
        "input_tokens": 5,
        "input_tokens_details": {"text_tokens": 2, "image_tokens": 3},
        "output_tokens": 4,
        "total_tokens": 9
    }});
    let catalog = catalog_with(&info);
    let cost = cost_calculator(request(&catalog, &response, &info)).unwrap();
    assert!((cost - (2.0 * 0.001 + 3.0 * 0.002 + 4.0 * 0.003)).abs() < 1e-12);
}

#[rstest]
fn azure_image_cost_uses_explicit_n_for_flat_image_rate() {
    let info = json!({"output_cost_per_image": 0.25, "input_cost_per_pixel": 0.0001});
    let response = json!({"data": [{}]});
    let catalog = catalog_with(&info);
    let cost = cost_calculator(AzureAiImageRequest {
        n: Some(3),
        ..request(&catalog, &response, &info)
    })
    .unwrap();
    assert_eq!(cost, 0.75);
}

#[rstest]
#[case(json!({"width": 20, "height": 10}), Some("4x4"), 200.0)]
#[case(json!({"width": 0, "height": 10}), Some("4x4"), 16.0)]
#[case(json!({}), None, 12.0)]
fn azure_pixel_cost_selects_request_or_response_dimensions(
    #[case] optional_params: Value,
    #[case] size: Option<&str>,
    #[case] pixels: f64,
) {
    let info = json!({"input_cost_per_pixel": 0.01});
    let response = json!({"data": [{}, {}], "size": "3x4"});
    let catalog = catalog_with(&info);
    let cost = cost_calculator(AzureAiImageRequest {
        size,
        optional_params: &optional_params,
        ..request(&catalog, &response, &info)
    })
    .unwrap();
    assert!((cost - pixels * 2.0 * 0.01).abs() < 1e-12);
}

#[rstest]
fn azure_image_cost_prefers_supplied_image_rate_before_shared_pixel_rate() {
    let shared = json!({"input_cost_per_pixel": 0.01});
    let supplied = json!({"input_cost_per_image": 0.2});
    let resolved = json!({"input_cost_per_pixel": 0.01, "input_cost_per_image": 0.2});
    let response = json!({"data": [{}, {}], "size": "3x4"});
    let catalog = catalog_with(&shared);
    let cost = cost_calculator(AzureAiImageRequest {
        catalog: &catalog,
        model: "azure_ai/model",
        image_response: &response,
        model_info: &resolved,
        supplied_model_info: Some(&supplied),
        shared_pricing: Some(&shared),
        size: None,
        n: None,
        optional_params: &Value::Null,
        at: at(),
    })
    .unwrap();
    assert_eq!(cost, 0.4);
    assert_eq!(input_cost_per_pixel(&supplied, Some(&shared)), 0.01);
}

#[rstest]
fn catalog_azure_image_cost_uses_deployment_pixel_rate_for_unlisted_model() {
    let catalog = ModelInfoCatalog::default();
    let response = json!({"data": [{}, {}], "size": "3x4"});
    let supplied = json!({"input_cost_per_pixel": 0.02});
    let cost = litellm_cost::image_cost_router::azure_ai_image_generation_cost(
        &catalog,
        AzureAiImageCatalogRequest {
            model: "unlisted",
            image_response: &response,
            size: None,
            n: None,
            optional_params: &json!({"width": 5, "height": 6}),
            supplied_model_info: Some(&supplied),
            at: at(),
        },
    )
    .unwrap();
    assert!((cost - 2.0 * 5.0 * 6.0 * 0.02).abs() < 1e-12);
}

#[rstest]
fn catalog_azure_image_cost_uses_shared_flat_rate_and_zero_when_unpriced() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "azure_ai/flat".to_owned(),
            json!({"output_cost_per_image": 0.3}),
        ),
        (
            "azure_ai/free".to_owned(),
            json!({"mode": "image_generation"}),
        ),
    ]));
    let response = json!({"data": [{}, {}]});
    let for_model = |model| AzureAiImageCatalogRequest {
        model,
        image_response: &response,
        size: None,
        n: None,
        optional_params: &Value::Null,
        supplied_model_info: None,
        at: at(),
    };
    assert_eq!(
        litellm_cost::image_cost_router::azure_ai_image_generation_cost(
            &catalog,
            for_model("flat")
        )
        .unwrap(),
        0.6
    );
    assert_eq!(
        litellm_cost::image_cost_router::azure_ai_image_generation_cost(
            &catalog,
            for_model("free")
        )
        .unwrap(),
        0.0
    );
}

#[rstest]
fn azure_pixel_pricing_prefers_a_size_specific_row_like_default_image_cost_calculator() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "azure_ai/flux-zz".to_owned(),
            json!({"input_cost_per_pixel": 1e-6}),
        ),
        (
            "azure_ai/1024-x-1024/flux-zz".to_owned(),
            json!({"input_cost_per_image": 0.3}),
        ),
    ]));
    let response = json!({"data": [{}]});
    let cost = litellm_cost::image_cost_router::azure_ai_image_generation_cost(
        &catalog,
        AzureAiImageCatalogRequest {
            model: "flux-zz",
            image_response: &response,
            size: Some("1024x1024"),
            n: None,
            optional_params: &Value::Null,
            supplied_model_info: None,
            at: at(),
        },
    )
    .unwrap();
    assert!((cost - 0.3).abs() < 1e-12);
}

#[rstest]
fn azure_pixel_pricing_bills_a_zero_dimension_as_zero_pixels() {
    let info = json!({"input_cost_per_pixel": 0.01});
    let catalog = catalog_with(&info);
    let response = json!({"data": [{}]});
    let cost = cost_calculator(AzureAiImageRequest {
        size: Some("1024x0"),
        ..request(&catalog, &response, &info)
    })
    .unwrap();
    assert_eq!(cost, 0.0);
}
