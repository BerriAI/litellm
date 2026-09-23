#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_usage_object_transformation.py::test_detects_interactions_usage_object

use litellm_cost::interactions_usage::{
    is_interactions_usage_object, transform_interactions_usage_object,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({"total_input_tokens": 1}), true)]
#[case(json!({"total_output_tokens": 1}), true)]
#[case(json!({"prompt_tokens": 1, "total_input_tokens": 1}), false)]
#[case(json!({"input_tokens": 1, "total_output_tokens": 1}), false)]
#[case(Value::Null, false)]
fn is_interactions_usage_object_distinguishes_api_shapes(
    #[case] usage: Value,
    #[case] expected: bool,
) {
    assert_eq!(is_interactions_usage_object(&usage), expected);
}

#[rstest]
fn transform_interactions_usage_object_prices_video_and_reasoning() {
    let raw = json!({
        "total_tokens": 18247,
        "total_input_tokens": 16,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 16}],
        "total_cached_tokens": 0,
        "total_output_tokens": 17937,
        "output_tokens_by_modality": [{"modality": "video", "tokens": 17376}],
        "total_tool_use_tokens": 0,
        "total_thought_tokens": 294
    });
    let usage = transform_interactions_usage_object(&raw).unwrap();
    assert_eq!(usage.prompt_tokens, 16);
    assert_eq!(usage.completion_tokens, 18231);
    assert_eq!(usage.total_tokens, 18247);
    assert_eq!(usage.prompt_tokens_details.unwrap().text_tokens, Some(16));
    let output = usage.completion_tokens_details.unwrap();
    assert_eq!(output.video_tokens, Some(17376));
    assert_eq!(output.reasoning_tokens, Some(294));
}

#[rstest]
#[case(5, 3, 25)]
#[case(0, 3, 23)]
fn transform_interactions_usage_object_prefers_reasoning_over_thoughts(
    #[case] reasoning: u64,
    #[case] thoughts: u64,
    #[case] expected_completion: u64,
) {
    let raw = json!({
        "total_input_tokens": 10,
        "total_output_tokens": 20,
        "total_reasoning_tokens": reasoning,
        "total_thought_tokens": thoughts
    });
    let usage = transform_interactions_usage_object(&raw).unwrap();
    assert_eq!(usage.completion_tokens, expected_completion);
    assert_eq!(usage.total_tokens, 10 + expected_completion);
}

#[rstest]
#[case(
    json!({"total_input_tokens": 1000, "input_tokens_by_modality": [{"modality": "text", "tokens": 1000}], "total_cached_tokens": 400, "total_output_tokens": 50}),
    Some(600),
    None,
    400
)]
#[case(
    json!({"total_input_tokens": 1500, "input_tokens_by_modality": [{"modality": "text", "tokens": 1000}, {"modality": "audio", "tokens": 500}], "total_cached_tokens": 300, "cached_tokens_by_modality": [{"modality": "audio", "tokens": 300}], "total_output_tokens": 50}),
    Some(1000),
    Some(200),
    300
)]
fn transform_interactions_usage_object_subtracts_cached_tokens_by_modality(
    #[case] raw: Value,
    #[case] text: Option<u64>,
    #[case] audio: Option<u64>,
    #[case] cached: u64,
) {
    let usage = transform_interactions_usage_object(&raw).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.text_tokens, text);
    assert_eq!(prompt.audio_tokens, audio);
    assert_eq!(prompt.cached_tokens, cached);
}

#[rstest]
fn transform_interactions_usage_object_adds_tool_use_tokens_to_input() {
    let raw = json!({
        "total_input_tokens": 100,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
        "total_tool_use_tokens": 40,
        "tool_use_tokens_by_modality": [{"modality": "text", "tokens": 40}],
        "total_output_tokens": 10
    });
    let usage = transform_interactions_usage_object(&raw).unwrap();
    assert_eq!(usage.prompt_tokens, 140);
    assert_eq!(usage.prompt_tokens_details.unwrap().text_tokens, Some(140));
}

#[rstest]
fn transform_interactions_usage_object_counts_google_search_requests() {
    let raw = json!({
        "total_input_tokens": 103,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 103}],
        "total_output_tokens": 226,
        "total_thought_tokens": 351,
        "grounding_tool_count": [
            {"type": "google_search", "count": 3},
            {"type": "url_context", "count": 2}
        ]
    });
    let usage = transform_interactions_usage_object(&raw).unwrap();
    assert_eq!(
        usage.prompt_tokens_details.unwrap().web_search_requests,
        Some(3)
    );
}

#[rstest]
fn transform_interactions_usage_object_folds_document_into_text() {
    let raw = json!({
        "total_input_tokens": 80,
        "input_tokens_by_modality": [
            {"modality": "text", "tokens": 30},
            {"modality": "document", "tokens": 50}
        ],
        "total_output_tokens": 10
    });
    let usage = transform_interactions_usage_object(&raw).unwrap();
    assert_eq!(usage.prompt_tokens_details.unwrap().text_tokens, Some(80));
}
