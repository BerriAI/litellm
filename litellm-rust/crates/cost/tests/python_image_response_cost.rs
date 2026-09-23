use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::image_response_cost::{
    calculate_image_response_cost_from_usage, calculate_image_response_web_search_cost,
    flat_image_cost, gemini_image_generation_cost, resolve_image_model_info,
    vertex_image_generation_cost,
};
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

#[rstest]
#[case("gemini", true)]
#[case("vertex_ai", true)]
#[case("openai", false)]
fn calculate_image_response_web_search_cost_uses_provider_and_billing_unit(
    #[case] provider: &str,
    #[case] supported: bool,
) {
    let response = json!({"usage": {"web_search_requests": 3}});
    let per_query = json!({
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {"search_context_size_medium": 0.02}
    });
    let per_prompt = json!({
        "web_search_billing_unit": "per_prompt",
        "search_context_cost_per_query": {"search_context_size_medium": 0.02}
    });
    let expected_query = if supported { 0.06 } else { 0.0 };
    let expected_prompt = if supported { 0.02 } else { 0.0 };
    assert!(
        (calculate_image_response_web_search_cost(&response, &per_query, provider)
            - expected_query)
            .abs()
            < 1e-12
    );
    assert!(
        (calculate_image_response_web_search_cost(&response, &per_prompt, provider)
            - expected_prompt)
            .abs()
            < 1e-12
    );
    assert_eq!(
        calculate_image_response_web_search_cost(&json!({"usage": {}}), &per_query, provider),
        0.0
    );
}

#[rstest]
#[case("gemini")]
#[case("vertex_ai")]
fn google_image_generation_prefers_token_usage_and_adds_web_search(#[case] provider: &str) {
    let info = json!({
        "input_cost_per_token": 0.001,
        "output_cost_per_image_token": 0.003,
        "output_cost_per_image": 0.5,
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {"search_context_size_medium": 0.02}
    });
    let response = json!({
        "data": [{"b64_json": "a"}, {"b64_json": "b"}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "web_search_requests": 2}
    });
    let direct = match provider {
        "gemini" => gemini_image_generation_cost(&response, &info, at()),
        _ => vertex_image_generation_cost(&response, &info, at()),
    };
    assert!((direct - (3.0 * 0.001 + 2.0 * 0.003 + 2.0 * 0.02)).abs() < 1e-12);
    let catalog = ModelInfoCatalog::new(HashMap::from([(format!("{provider}/image-model"), info)]));
    assert_eq!(
        catalog
            .google_image_generation_cost("image-model", provider, &response, None, at())
            .unwrap(),
        direct
    );
}

#[rstest]
fn google_image_generation_falls_back_to_images_and_preserves_supplied_prices() {
    let shared = json!({"output_cost_per_image": 0.5, "web_search_billing_unit": "per_prompt"});
    let supplied = json!({"output_cost_per_image": 0.25, "search_context_cost_per_query": {"search_context_size_medium": 0.02}});
    let merged = resolve_image_model_info(Some(&shared), Some(&supplied)).unwrap();
    assert_eq!(merged["output_cost_per_image"], json!(0.25));
    assert_eq!(merged["web_search_billing_unit"], json!("per_prompt"));
    let response = json!({
        "data": [{"b64_json": "a"}, {"b64_json": "b"}],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "web_search_requests": 3}
    });
    assert_eq!(flat_image_cost(&response, &merged), 0.5);
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/image-model".to_owned(),
        shared,
    )]));
    let cost = catalog
        .google_image_generation_cost("image-model", "vertex_ai", &response, Some(&supplied), at())
        .unwrap();
    assert!((cost - 0.52).abs() < 1e-12);
}
