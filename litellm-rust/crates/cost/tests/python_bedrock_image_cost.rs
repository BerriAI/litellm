#![allow(clippy::disallowed_types)]

// mirrors: image_gen_tests/test_bedrock_image_gen_unit_tests.py::test_cost_calculator_stability1
// expected values recorded from litellm/llms/bedrock/image_generation/cost_calculator.py::cost_calculator

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::bedrock_image_cost::{
    BedrockImageFamily, cost_calculator, get_config_class, stability1_pricing_key,
};
use litellm_cost::call_type::CallTypes;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::error::CostError;
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
#[case::at_the_threshold(json!({"steps": 50}), "50-steps")]
#[case::above_the_threshold(json!({"steps": 51}), "max-steps")]
#[case::float_steps(json!({"steps": 50.5}), "max-steps")]
#[case::bool_steps_compare_as_int(json!({"steps": true}), "50-steps")]
#[case::default_steps(json!({}), "50-steps")]
fn stability1_pricing_key_selects_step_tier(#[case] params: Value, #[case] tier: &str) {
    assert_eq!(
        stability1_pricing_key("stability.sd-zz-v1", Some("1024-x-1024"), &params),
        Ok(format!("1024-x-1024/{tier}/stability.sd-zz-v1"))
    );
}

fn bedrock_image_catalog() -> ModelInfoCatalog {
    let entry = |rate: f64| json!({"output_cost_per_image": rate, "litellm_provider": "bedrock", "mode": "image_generation"});
    ModelInfoCatalog::new(HashMap::from([
        (
            "1024-x-1024/50-steps/stability.sd-zz-v1".to_owned(),
            entry(0.04),
        ),
        (
            "1024-x-1024/max-steps/stability.sd-zz-v1".to_owned(),
            entry(0.08),
        ),
        ("amazon.nova-canvas-zz".to_owned(), entry(0.03)),
        ("amazon.titan-image-zz".to_owned(), entry(0.02)),
        ("stability.sd3-zz".to_owned(), entry(0.05)),
    ]))
}

#[rstest]
#[case::stability1_default_tier("stability.sd-zz-v1", Some("1024-x-1024"), json!({"steps": 50}), Ok(0.08))]
#[case::stability1_max_tier("stability.sd-zz-v1", Some("1024-x-1024"), json!({"steps": 60}), Ok(0.16))]
#[case::stability1_missing_steps("stability.sd-zz-v1", Some("1024-x-1024"), json!({}), Ok(0.08))]
#[case::stability1_default_size("stability.sd-zz-v1", None, json!({"steps": true}), Ok(0.08))]
#[case::stability1_empty_size_uses_the_default("stability.sd-zz-v1", Some(""), json!({}), Ok(0.08))]
#[case::stability1_null_steps("stability.sd-zz-v1", Some("1024-x-1024"), json!({"steps": null}), Err(CostError::InvalidSteps))]
#[case::stability1_string_steps("stability.sd-zz-v1", Some("1024-x-1024"), json!({"steps": "60"}), Err(CostError::InvalidSteps))]
#[case::stability1_keeps_the_routing_prefix_in_the_key(
    "bedrock/stability.sd-zz-v1",
    Some("1024-x-1024"),
    json!({"steps": 50}),
    Err(CostError::ModelNotFound)
)]
#[case::stability1_key_is_case_insensitive("STABILITY.SD-ZZ-V1", Some("1024-x-1024"), json!({}), Ok(0.08))]
#[case::nova_bare("amazon.nova-canvas-zz", None, json!({}), Ok(0.06))]
#[case::nova_routing_prefix("bedrock/amazon.nova-canvas-zz", None, json!({}), Ok(0.06))]
#[case::nova_cross_region_prefix("us.amazon.nova-canvas-zz", None, json!({}), Ok(0.06))]
#[case::nova_region_path("bedrock/us-east-1/amazon.nova-canvas-zz", None, json!({}), Ok(0.06))]
#[case::nova_invoke_route("invoke/amazon.nova-canvas-zz", None, json!({}), Ok(0.06))]
#[case::nova_unmapped("amazon.nova-canvas-missing", None, json!({}), Err(CostError::ModelNotFound))]
#[case::titan_bare("amazon.titan-image-zz", None, json!({}), Ok(0.04))]
#[case::titan_routing_prefix("bedrock/amazon.titan-image-zz", None, json!({}), Ok(0.04))]
#[case::titan_infers_no_provider_for_a_cross_region_id(
    "us.amazon.titan-image-zz",
    None,
    json!({}),
    Err(CostError::ModelNotFound)
)]
#[case::stability3_bare("stability.sd3-zz", None, json!({}), Ok(0.1))]
#[case::stability3_routing_prefix("bedrock/stability.sd3-zz", None, json!({}), Ok(0.1))]
fn bedrock_image_cost_resolves_models_like_get_model_info(
    #[case] model: &str,
    #[case] size: Option<&str>,
    #[case] params: Value,
    #[case] expected: Result<f64, CostError>,
) {
    let response = json!({"data": [{"url": "x"}, {"url": "y"}]});
    assert_eq!(
        cost_calculator(&bedrock_image_catalog(), model, &response, size, &params),
        expected
    );
}

#[rstest]
#[case("amazon.titan-image-generator-v2", 0.02)]
#[case("amazon.nova-canvas-v1:0", 0.03)]
#[case("stability.sd3-large-v1:0", 0.04)]
fn bedrock_direct_families_bill_returned_images(#[case] model: &str, #[case] rate: f64) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        model.to_owned(),
        json!({"output_cost_per_image": rate, "litellm_provider": "bedrock"}),
    )]));
    let response = json!({"data": [{}, {}, {}]});
    let supplied = json!({"output_cost_per_image": 1.0});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            model,
            provider: Some("bedrock"),
            image_response: &response,
            call_type: Some(CallTypes::image_generation),
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
