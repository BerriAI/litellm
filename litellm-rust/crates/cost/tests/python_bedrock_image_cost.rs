#![allow(clippy::disallowed_types)]

// mirrors: image_gen_tests/test_bedrock_image_gen_unit_tests.py::test_cost_calculator_stability1

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::bedrock_image_cost::{
    BedrockImageFamily, cost_calculator, get_config_class, stability1_pricing_key,
};
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::image_cost_router::{
    ImageCostRouteRequest, route_image_generation_cost_calculator,
};
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-09-22T12:00:00Z".parse().unwrap()
}

#[rstest]
#[case("bedrock/amazon.titan-image-generator-v2", BedrockImageFamily::Titan)]
#[case("bedrock/amazon.nova-canvas-v1:0", BedrockImageFamily::NovaCanvas)]
#[case("bedrock/stability.sd3-large-v1:0", BedrockImageFamily::Stability3)]
#[case(
    "bedrock/stability.stable-image-ultra-v1:0",
    BedrockImageFamily::Stability3
)]
#[case(
    "bedrock/stability.stable-diffusion-xl-v1",
    BedrockImageFamily::Stability1
)]
fn get_config_class_matches_model_family(
    #[case] model: &str,
    #[case] expected: BedrockImageFamily,
) {
    assert_eq!(get_config_class(model), expected);
}

#[rstest]
#[case(50, "50-steps")]
#[case(51, "max-steps")]
fn stability1_pricing_key_selects_step_tier(#[case] steps: u64, #[case] tier: &str) {
    let key = stability1_pricing_key(
        "stability.stable-diffusion-xl-v1",
        Some("1024-x-1024"),
        &json!({"steps": steps}),
    );
    assert_eq!(
        key,
        format!("1024-x-1024/{tier}/stability.stable-diffusion-xl-v1")
    );
}

#[rstest]
fn bedrock_stability1_cost_uses_size_and_steps_lookup() {
    let entries = HashMap::from([
        (
            "1024-x-1024/50-steps/stability.stable-diffusion-xl-v1".to_owned(),
            json!({"output_cost_per_image": 0.04}),
        ),
        (
            "1024-x-1024/max-steps/stability.stable-diffusion-xl-v1".to_owned(),
            json!({"output_cost_per_image": 0.08}),
        ),
    ]);
    let response = json!({"data": [{}, {}]});
    assert_eq!(
        cost_calculator(
            "bedrock/stability.stable-diffusion-xl-v1",
            &response,
            Some("1024-x-1024"),
            &json!({"steps": 50}),
            &entries
        ),
        Some(0.08)
    );
    assert_eq!(
        cost_calculator(
            "bedrock/stability.stable-diffusion-xl-v1",
            &response,
            Some("1024-x-1024"),
            &json!({"steps": 60}),
            &entries
        ),
        Some(0.16)
    );
}

#[rstest]
#[case("amazon.titan-image-generator-v2", 0.02)]
#[case("amazon.nova-canvas-v1:0", 0.03)]
#[case("stability.sd3-large-v1:0", 0.04)]
fn bedrock_direct_families_bill_returned_images(#[case] model: &str, #[case] rate: f64) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("bedrock/{model}"),
        json!({"output_cost_per_image": rate}),
    )]));
    let response = json!({"data": [{}, {}, {}]});
    let supplied = json!({"output_cost_per_image": 1.0});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            model,
            provider: Some("bedrock"),
            image_response: &response,
            call_type: Some("image_generation"),
            quality: None,
            size: Some("1024-x-1024"),
            n: None,
            optional_params: &Value::Null,
            supplied_model_info: Some(&supplied),
            at: at(),
        },
    )
    .unwrap();
    assert!((cost - rate * 3.0).abs() < 1e-12);
}
