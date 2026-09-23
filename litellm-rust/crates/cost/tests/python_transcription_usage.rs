#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_usage_object_transformation.py

use litellm_cost::transcription_usage::{
    is_transcription_usage_object, transform_transcription_usage_object,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({"type": "tokens", "input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "input_token_details": {"text_tokens": 4, "audio_tokens": 16}}), true)]
#[case(json!({"type": "duration", "seconds": 4.5}), true)]
#[case(json!({"type": "unknown", "seconds": 4.5}), false)]
#[case(json!({"type": "tokens", "input_tokens": 20}), false)]
fn is_transcription_usage_object_recognizes_tagged_shapes(
    #[case] usage: Value,
    #[case] expected: bool,
) {
    assert_eq!(is_transcription_usage_object(&usage), expected);
}

#[rstest]
fn transform_transcription_usage_object_maps_token_details() {
    let raw = json!({
        "type": "tokens",
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "input_token_details": {"text_tokens": 4, "audio_tokens": 16}
    });
    let usage = transform_transcription_usage_object(&raw).unwrap().unwrap();
    assert_eq!(usage.prompt_tokens, 20);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 25);
    let details = usage.prompt_tokens_details.unwrap();
    assert_eq!(details.text_tokens, Some(4));
    assert_eq!(details.audio_tokens, Some(16));
}

#[rstest]
fn transform_transcription_usage_object_discards_duration_billing() {
    let response = json!({"usage": {"type": "duration", "seconds": 4.5}});
    assert_eq!(get_usage_object(&response), Ok(None));
}

#[rstest]
fn get_usage_object_dispatches_transcription_tokens() {
    let response = json!({"usage": {
        "type": "tokens",
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "input_token_details": {"text_tokens": 4, "audio_tokens": 16}
    }});
    let usage = get_usage_object(&response).unwrap().unwrap();
    assert_eq!(usage.prompt_tokens_details.unwrap().audio_tokens, Some(16));
}
