#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py
use litellm_cost::error::CostError;

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::responses_usage::ChatUsage;
use litellm_cost::speech_cost::{
    SpeechCostMetric, cost_per_second, count_characters, generic_cost_per_character,
    lyria_generation_cost, select_cost_metric_for_model, transcription_usage_has_token_details,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn request<'a>(model: &'a str, provider: &'a str, usage: &'a ChatUsage) -> ModelCostRequest<'a> {
    ModelCostRequest {
        model,
        provider: Some(provider),
        region: None,
        usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-09-22T12:00:00Z".parse::<Timestamp>().unwrap(),
        response_time_ms: None,
    }
}

#[rstest]
#[case("hello world", 10)]
#[case(" a\t雪\u{2003}b ", 3)]
#[case("a\u{001c}b", 2)]
fn count_characters_excludes_whitespace(#[case] prompt: &str, #[case] expected: usize) {
    assert_eq!(count_characters(prompt), expected);
}

#[rstest]
#[case(json!({"input_cost_per_character": 0.002, "input_cost_per_token": 0.003}), Ok(SpeechCostMetric::PerCharacter))]
#[case(json!({"input_cost_per_character": 0.0, "input_cost_per_token": 0.003}), Ok(SpeechCostMetric::PerToken))]
#[case(json!({"input_cost_per_character": 0.0, "input_cost_per_token": 0.0}), Err(CostError::MissingMetric))]
fn select_cost_metric_for_model_uses_character_first(
    #[case] model_info: Value,
    #[case] expected: Result<SpeechCostMetric, CostError>,
) {
    assert_eq!(select_cost_metric_for_model(&model_info), expected);
}

#[rstest]
fn generic_cost_per_character_uses_custom_rates_and_preserves_missing_rate() {
    let info = json!({"input_cost_per_character": 0.002});
    assert_eq!(
        generic_cost_per_character(&info, 10.0, 4.0, None, None),
        (Some(0.02), None)
    );
    assert_eq!(
        generic_cost_per_character(&info, 10.0, 4.0, Some(0.003), Some(0.005)),
        (Some(0.03), Some(0.02))
    );
}

#[rstest]
#[case(json!({"input_cost_per_second": 0.002, "output_cost_per_second": 0.003}), (0.0, 0.03))]
#[case(json!({"input_cost_per_second": 0.002, "output_cost_per_second": 0.0}), (0.02, 0.0))]
#[case(json!({"output_cost_per_second": 0.003}), (0.0, 0.03))]
fn cost_per_second_bills_one_side_only(#[case] info: Value, #[case] expected: (f64, f64)) {
    let actual = cost_per_second(&info, 10.0);
    assert!((actual.0 - expected.0).abs() < 1e-12);
    assert!((actual.1 - expected.1).abs() < 1e-12);
}

#[rstest]
fn speech_cost_selects_character_or_token_pricing_and_requires_characters() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/character".to_owned(),
            json!({"input_cost_per_character": 0.002, "input_cost_per_token": 0.003, "output_cost_per_token": 0.004}),
        ),
        (
            "openai/token".to_owned(),
            json!({"input_cost_per_token": 0.003, "output_cost_per_token": 0.004}),
        ),
    ]));
    let usage = get_usage_object(&json!({"usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
        .unwrap()
        .unwrap();
    assert_eq!(
        litellm_cost::cost_calculator::speech_cost(
            &catalog,
            request("character", "openai", &usage),
            Some(5.0)
        )
        .unwrap(),
        (0.01, 0.0)
    );
    assert_eq!(
        litellm_cost::cost_calculator::speech_cost(
            &catalog,
            request("token", "openai", &usage),
            None
        )
        .unwrap(),
        (0.03, 0.008)
    );
    assert_eq!(
        litellm_cost::cost_calculator::speech_cost(
            &catalog,
            request("character", "openai", &usage),
            None
        ),
        Err(CostError::MissingPromptCharacters)
    );
}

#[rstest]
fn speech_cost_uses_lyria_generation_price_before_character_or_token_rates() {
    let info = json!({"vertex_ai_audio_api": "lyria_predict", "supported_audio_formats": ["wav"], "output_cost_per_image": 0.08});
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/lyria".to_owned(),
        info.clone(),
    )]));
    assert_eq!(lyria_generation_cost(&info), Some(0.08));
    assert_eq!(
        litellm_cost::cost_calculator::speech_cost(
            &catalog,
            request("lyria", "vertex_ai", &ChatUsage::default()),
            None
        )
        .unwrap(),
        (0.0, 0.08)
    );
    assert_eq!(
        lyria_generation_cost(&json!({"output_cost_per_image": 0.08})),
        None
    );
}

#[rstest]
fn transcription_cost_selects_token_usage_or_duration_pricing() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/transcribe".to_owned(),
        json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003, "input_cost_per_second": 0.01, "output_cost_per_second": 0.0}),
    )]));
    let token_usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
            .unwrap()
            .unwrap();
    let duration_usage = ChatUsage::default();
    let audio_detail_usage =
        get_usage_object(&json!({"usage": {"prompt_tokens_details": {"audio_tokens": 3}}}))
            .unwrap()
            .unwrap();
    assert!(transcription_usage_has_token_details(&token_usage));
    assert!(transcription_usage_has_token_details(&audio_detail_usage));
    assert!(!transcription_usage_has_token_details(&duration_usage));
    assert_eq!(
        litellm_cost::cost_calculator::transcription_cost(
            &catalog,
            request("transcribe", "openai", &token_usage),
            5.0
        )
        .unwrap(),
        (0.02, 0.006)
    );
    assert_eq!(
        litellm_cost::cost_calculator::transcription_cost(
            &catalog,
            request("transcribe", "openai", &duration_usage),
            5.0
        )
        .unwrap(),
        (0.05, 0.0)
    );
}

#[rstest]
fn speech_priced_per_token_uses_generic_rates_not_the_provider_per_second_route() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "azure/tts".to_owned(),
        json!({"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "output_cost_per_second": 0.01}),
    )]));
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}))
            .unwrap()
            .unwrap();
    let (prompt, completion) = litellm_cost::cost_calculator::speech_cost(
        &catalog,
        ModelCostRequest {
            response_time_ms: Some(1000.0),
            ..request("tts", "azure", &usage)
        },
        None,
    )
    .unwrap();
    assert!((prompt - 1e-3).abs() < 1e-15);
    assert!((completion - 2e-3).abs() < 1e-15);
}

#[rstest]
fn transcription_token_pricing_ignores_the_vertex_location() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/transcribe".to_owned(),
        json!({"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "regional_endpoint_uplift_multiplier": 3.0}),
    )]));
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}))
            .unwrap()
            .unwrap();
    let (prompt, completion) = litellm_cost::cost_calculator::transcription_cost(
        &catalog,
        ModelCostRequest {
            vertex_location: Some("us-east5"),
            ..request("transcribe", "openai", &usage)
        },
        0.0,
    )
    .unwrap();
    assert!((prompt - 1e-3).abs() < 1e-15);
    assert!((completion - 2e-3).abs() < 1e-15);
}

#[rstest]
#[case::vertex_ai("vertex_ai", "lyria")]
#[case::vertex_ai_beta_reads_the_vertex_ai_key("vertex_ai_beta", "lyria")]
#[case::prefixed_model("vertex_ai", "vertex_ai/lyria")]
fn speech_cost_reads_lyria_prices_from_the_vertex_ai_key(
    #[case] provider: &str,
    #[case] model: &str,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/lyria".to_owned(),
        json!({"vertex_ai_audio_api": "lyria_predict", "supported_audio_formats": ["wav"], "output_cost_per_image": 0.08}),
    )]));
    assert_eq!(
        litellm_cost::cost_calculator::speech_cost(
            &catalog,
            request(model, provider, &ChatUsage::default()),
            None
        )
        .unwrap(),
        (0.0, 0.08)
    );
}
