use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::image_response_cost::calculate_image_response_cost_from_usage;
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-01-01T12:00Z".parse().unwrap()
}

fn model_info() -> Value {
    json!({
        "input_cost_per_token": 5e-6,
        "input_cost_per_image_token": 8e-6,
        "output_cost_per_token": 7e-6,
        "output_cost_per_image_token": 3e-5
    })
}

#[rstest]
#[case(json!({"image_tokens": 158, "text_tokens": 0}), 19.0 * 5e-6 + 512.0 * 8e-6 + 158.0 * 3e-5)]
#[case(json!({"image_tokens": 100}), 19.0 * 5e-6 + 512.0 * 8e-6 + 100.0 * 3e-5 + 58.0 * 7e-6)]
#[case(Value::Null, 19.0 * 5e-6 + 512.0 * 8e-6 + 158.0 * 3e-5)]
fn calculate_image_response_cost_from_usage_prices_image_input_and_output_details(
    #[case] output_details: Value,
    #[case] expected: f64,
) {
    let response = json!({"usage": {
        "input_tokens": 531,
        "output_tokens": 158,
        "total_tokens": 689,
        "input_tokens_details": {"text_tokens": 19, "image_tokens": 512},
        "output_tokens_details": output_details
    }});
    let direct =
        calculate_image_response_cost_from_usage(&response, &model_info(), Some("openai"), at());
    assert!((direct.unwrap() - expected).abs() < 1e-12);
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/image-model".to_owned(),
        model_info(),
    )]));
    assert_eq!(
        catalog.calculate_image_response_cost_from_usage(
            "image-model",
            Some("openai"),
            None,
            &response,
            at(),
        ),
        direct
    );
}

#[rstest]
#[case(json!({}), true)]
#[case(json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}), true)]
#[case(json!({"input_tokens": 1, "output_tokens": 1}), true)]
#[case(json!({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}), false)]
fn calculate_image_response_cost_from_usage_requires_real_complete_token_usage(
    #[case] usage: Value,
    #[case] falls_back: bool,
) {
    let response = json!({"usage": usage});
    let cost =
        calculate_image_response_cost_from_usage(&response, &model_info(), Some("openai"), at());
    assert_eq!(cost.is_none(), falls_back);
    if !falls_back {
        assert!((cost.unwrap() - (5e-6 + 3e-5)).abs() < 1e-12);
    }
}

#[rstest]
fn calculate_image_response_cost_from_usage_falls_back_when_model_does_not_price_tokens() {
    let response = json!({"usage": {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2
    }});
    let model_info = json!({"output_cost_per_image": 0.01});
    assert_eq!(
        calculate_image_response_cost_from_usage(&response, &model_info, Some("openai"), at()),
        None
    );
}
