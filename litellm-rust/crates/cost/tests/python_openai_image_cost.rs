#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_gpt_image_cost_calculator.py

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::call_type::CallTypes;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::image_cost_router::{
    ImageCostRouteRequest, route_image_generation_cost_calculator,
};
use litellm_cost::openai_image_cost::cost_calculator;
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-09-22T12:00:00Z".parse().unwrap()
}

fn info() -> Value {
    json!({
        "input_cost_per_token": 0.001,
        "input_cost_per_image_token": 0.002,
        "output_cost_per_token": 0.004,
        "output_cost_per_image_token": 0.003,
        "output_cost_per_image": 0.5
    })
}

#[rstest]
fn openai_image_cost_uses_chat_output_details_before_image_usage_fields() {
    let response = json!({"data": [{}], "usage": {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
        "completion_tokens_details": {"text_tokens": 0, "image_tokens": 5},
        "input_tokens": 100,
        "output_tokens": 100
    }});
    let cost = cost_calculator(&response, &info(), "openai", at()).unwrap();
    assert!((cost - (3.0 * 0.001 + 5.0 * 0.003)).abs() < 1e-12);
}

#[rstest]
fn openai_image_cost_prices_image_usage_with_split_input_details() {
    let response = json!({"data": [{}], "usage": {
        "input_tokens": 5,
        "input_tokens_details": {"text_tokens": 3, "image_tokens": 2},
        "output_tokens": 4,
        "total_tokens": 9
    }});
    let cost = cost_calculator(&response, &info(), "openai", at()).unwrap();
    assert!((cost - (3.0 * 0.001 + 2.0 * 0.002 + 4.0 * 0.003)).abs() < 1e-12);
}

#[rstest]
fn openai_image_cost_prices_chat_usage_without_output_breakdown_as_text() {
    let response = json!({"data": [{}], "usage": {
        "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8
    }});
    let cost = cost_calculator(&response, &info(), "openai", at()).unwrap();
    assert!((cost - (3.0 * 0.001 + 5.0 * 0.004)).abs() < 1e-12);
}

#[rstest]
#[case(json!({"data": [{}, {}]}))]
#[case(json!({"data": [{}, {}], "usage": {"input_tokens": 1, "output_tokens": 2}}))]
fn openai_image_cost_falls_back_to_returned_images_when_usage_unpriced(#[case] response: Value) {
    assert_eq!(
        cost_calculator(&response, &info(), "openai", at()).unwrap(),
        1.0
    );
}

#[rstest]
#[case("openai")]
#[case("azure")]
fn image_router_uses_gpt_image_path_and_deployment_flat_rate(#[case] provider: &str) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("{provider}/gpt-image-model"),
        info(),
    )]));
    let response = json!({"data": [{}, {}]});
    let supplied = json!({"output_cost_per_image": "0.25"});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            model: "gpt-image-model",
            provider: Some(provider),
            image_response: &response,
            call_type: Some(CallTypes::image_generation),
            quality: None,
            size: None,
            n: None,
            optional_params: &Value::Null,
            supplied_model_info: Some(&supplied),
            at: at(),
        },
    )
    .unwrap();
    assert_eq!(cost, 0.5);
}
