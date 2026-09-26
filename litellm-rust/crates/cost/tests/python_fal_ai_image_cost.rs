#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/fal_ai/test_cost_calculator.py

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::call_type::CallTypes;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::fal_ai_image_cost::{
    PIXELS_PER_MEGAPIXEL, cost_calculator, fal_ai_passthrough_cost,
    fal_ai_passthrough_cost_from_model_info, flat_cost_per_image, image_dimensions,
    keyed_cost_per_image, keyed_quality, keyed_rows, keyed_size, parse_keyed_dimensions,
};
use litellm_cost::image_cost_router::{
    ImageCostRouteRequest, route_image_generation_cost_calculator,
};
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-09-22T12:00:00Z".parse().unwrap()
}

fn image(width: Value, height: Value) -> Value {
    json!({"provider_specific_fields": {"width": width, "height": height}})
}

#[rstest]
#[case(json!({}), Some((1024, 768)))]
#[case(json!({"image_size": "auto"}), Some((1024, 768)))]
#[case(json!({"image_size": "square_hd"}), Some((1024, 1024)))]
#[case(json!({"image_size": {"width": 800, "height": 600}}), Some((800, 600)))]
#[case(json!({"image_size": "unknown"}), None)]
fn keyed_size_maps_request_shapes(#[case] params: Value, #[case] expected: Option<(u64, u64)>) {
    assert_eq!(keyed_size(&params), expected);
}

#[rstest]
fn image_dimensions_require_positive_integer_provider_fields() {
    assert_eq!(
        image_dimensions(&image(json!(1024), json!(768))),
        Some((1024, 768))
    );
    for invalid in [
        image(json!(true), json!(768)),
        image(json!(0), json!(768)),
        image(json!(-1), json!(768)),
    ] {
        assert_eq!(image_dimensions(&invalid), None);
    }
    assert_eq!(parse_keyed_dimensions("1024-x-768"), Some((1024, 768)));
    assert_eq!(parse_keyed_dimensions("1024x768"), None);
    assert_eq!(keyed_quality(&json!({"quality": "auto"})), "high");
}

#[rstest]
fn keyed_rows_price_response_dimensions_and_choose_nearest_area() {
    let entries = HashMap::from([
        (
            "fal_ai/low/1024-x-768/openai/model".to_owned(),
            json!({"output_cost_per_image": 0.10}),
        ),
        (
            "fal_ai/low/1024-x-1536/openai/model".to_owned(),
            json!({"output_cost_per_image": 0.20}),
        ),
    ]);
    let rows = keyed_rows(&entries, "openai/model", "low");
    assert_eq!(rows.len(), 2);
    let request = json!({"quality": "low", "image_size": {"width": 1024, "height": 768}});
    assert_eq!(
        keyed_cost_per_image(&rows, &image(json!(1024), json!(1536)), &request),
        Some(0.20)
    );
    assert_eq!(
        keyed_cost_per_image(&rows, &image(json!(777), json!(888)), &request),
        Some(0.10)
    );
}

#[rstest]
fn flat_cost_per_image_rounds_each_response_to_megapixels() {
    let rate = 0.0001;
    let images = [(1024, 1024), (1920, 1080), (512, 512)];
    let total: f64 = images
        .into_iter()
        .map(|(width, height)| {
            flat_cost_per_image(&image(json!(width), json!(height)), 0.5, Some(rate))
        })
        .sum();
    assert!((total - rate * PIXELS_PER_MEGAPIXEL as f64 * 4.0).abs() < 1e-12);
    assert_eq!(
        flat_cost_per_image(&image(json!(true), json!(768)), 0.5, Some(rate)),
        0.5
    );
}

#[rstest]
fn fal_image_cost_uses_deployment_rate_before_keyed_and_base_prices() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "fal_ai/high/1024-x-768/openai/model".to_owned(),
            json!({"output_cost_per_image": 0.10}),
        ),
        (
            "fal_ai/openai/model".to_owned(),
            json!({"output_cost_per_image": 0.50}),
        ),
    ]));
    let response = json!({"data": [{}, {}]});
    let supplied = json!({"output_cost_per_image": 0.25});
    assert_eq!(
        cost_calculator(
            &catalog,
            "fal_ai/openai/model",
            &response,
            &json!({}),
            Some(&supplied),
        ),
        Some(0.50)
    );
    assert_eq!(
        cost_calculator(&catalog, "fal_ai/openai/model", &response, &json!({}), None),
        Some(0.20)
    );
}

#[rstest]
fn fal_image_cost_falls_back_to_base_megapixel_and_flat_rates_per_image() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "fal_ai/fal-ai/flux/dev".to_owned(),
        json!({"output_cost_per_image": 0.5, "output_cost_per_pixel": 0.0001}),
    )]));
    let response = json!({"data": [
        image(json!(1024), json!(1024)),
        image(json!(1920), json!(1080)),
        image(json!(true), json!(1024))
    ]});
    let cost = cost_calculator(
        &catalog,
        "fal_ai/fal-ai/flux/dev",
        &response,
        &json!({}),
        None,
    )
    .unwrap();
    let expected = 0.0001 * PIXELS_PER_MEGAPIXEL as f64 * 3.0 + 0.5;
    assert!((cost - expected).abs() < 1e-12);
}

#[rstest]
fn catalog_fal_image_route_uses_keyed_row_for_image_edit() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "fal_ai/medium/1024-x-1024/openai/model".to_owned(),
        json!({"output_cost_per_image": 0.15}),
    )]));
    let response = json!({"data": [{}]});
    let params = json!({"quality": "medium", "image_size": {"width": 1024, "height": 1024}});
    let cost = route_image_generation_cost_calculator(
        &catalog,
        ImageCostRouteRequest {
            model: "openai/model",
            provider: Some("fal_ai"),
            image_response: &response,
            call_type: Some(CallTypes::aimage_edit),
            quality: None,
            size: None,
            n: None,
            optional_params: &params,
            supplied_model_info: None,
            at: at(),
        },
    )
    .unwrap();
    assert_eq!(cost, 0.15);
}

#[rstest]
fn fal_passthrough_selects_resolution_rate_and_falls_back_to_base() {
    let info = json!({"output_cost_per_image": 0.3, "output_cost_per_image_512": 0.25});
    assert_eq!(
        fal_ai_passthrough_cost_from_model_info(Some(&info), &json!({"resolution": 512})),
        Some(0.25)
    );
    assert_eq!(
        fal_ai_passthrough_cost_from_model_info(Some(&info), &json!({"resolution": "512"})),
        Some(0.25)
    );
    assert_eq!(
        fal_ai_passthrough_cost_from_model_info(Some(&info), &json!({"resolution": true})),
        Some(0.3)
    );
    assert_eq!(
        fal_ai_passthrough_cost_from_model_info(Some(&info), &json!({"resolution": 512.0})),
        Some(0.3)
    );
    let catalog =
        ModelInfoCatalog::new(HashMap::from([("fal_ai/fal-ai/trellis".to_owned(), info)]));
    assert_eq!(
        fal_ai_passthrough_cost(&catalog, "fal-ai/trellis", &json!({"resolution": 512})),
        Some(0.25)
    );
    assert_eq!(
        fal_ai_passthrough_cost(
            &catalog,
            "fal_ai/fal-ai/trellis",
            &json!({"resolution": 512})
        ),
        None
    );
}

// expected values recorded from litellm/llms/fal_ai/cost_calculator.py::cost_calculator
#[rstest]
#[case::bare_key_owned_by_fal("fal_ai/flux-zz", Some(0.4))]
#[case::case_insensitive_key("FLUX-ZZ", Some(0.4))]
#[case::bool_rate_counts_as_int("fal_ai/flag-zz", Some(2.0))]
#[case::string_rate_is_not_a_number("fal_ai/str-zz", Some(0.0))]
#[case::unmapped_model("fal_ai/missing", None)]
fn fal_image_flat_fallback_resolves_through_get_model_info(
    #[case] model: &str,
    #[case] expected: Option<f64>,
) {
    let entry = |rate: Value| json!({"output_cost_per_image": rate, "litellm_provider": "fal_ai"});
    let catalog = ModelInfoCatalog::new(HashMap::from([
        ("flux-zz".to_owned(), entry(json!(0.2))),
        ("fal_ai/flag-zz".to_owned(), entry(json!(true))),
        ("fal_ai/str-zz".to_owned(), entry(json!("0.3"))),
    ]));
    let response = json!({"data": [{"url": "x"}, {"url": "y"}]});
    assert_eq!(
        cost_calculator(
            &catalog,
            model,
            &response,
            &json!({"image_size": "weird"}),
            None
        ),
        expected
    );
}
