use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::realtime_cost::{
    get_transcription_model_name_from_results, transcription_usage_cost,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({"audio": {"input": {"transcription": {"model": "audio"}}}, "model": "session"}), "audio")]
#[case(json!({"input_audio_transcription": {"model": "legacy"}, "model": "session"}), "legacy")]
#[case(json!({"audio": {"input": {"transcription": {}}}, "input_audio_transcription": {"model": "legacy"}, "model": "session"}), "legacy")]
#[case(json!({"audio": {"input": {"transcription": {"model": ""}}}, "input_audio_transcription": {"model": "legacy"}, "model": "session"}), "session")]
#[case(json!({"model": "session"}), "session")]
fn get_transcription_model_name_from_results_uses_session_precedence(
    #[case] session: Value,
    #[case] expected: &str,
) {
    let events = [
        json!({"type": "response.done"}),
        json!({"type": "session.updated", "session": session}),
    ];
    assert_eq!(
        get_transcription_model_name_from_results(&events),
        Some(expected)
    );
}

#[rstest]
#[case(json!({"type": "duration", "seconds": 2.5}), 2.5 * 0.02)]
#[case(json!({"type": "tokens", "input_token_details": {"audio_tokens": 3, "text_tokens": 4}, "output_tokens": 2}), 3.0 * 0.04 + 4.0 * 0.01 + 2.0 * 0.03)]
#[case(json!({"type": "future"}), 0.0)]
fn transcription_usage_cost_prices_supported_units(#[case] usage: Value, #[case] expected: f64) {
    let prices = json!({
        "input_cost_per_second": 0.02,
        "input_cost_per_audio_token": 0.04,
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.03
    });
    assert!((transcription_usage_cost(&usage, Some(&prices)) - expected).abs() < 1e-12);
    assert_eq!(transcription_usage_cost(&usage, None), 0.0);
}

#[rstest]
fn transcription_usage_cost_falls_back_to_input_rate_for_audio() {
    let prices = json!({"input_cost_per_token": 0.01});
    let usage = json!({"type": "tokens", "input_token_details": {"audio_tokens": 3}});
    assert_eq!(transcription_usage_cost(&usage, Some(&prices)), 3.0 * 0.01);
}

#[rstest]
fn handle_realtime_transcription_cost_calculation_uses_session_model_and_completed_events() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/asr".to_owned(),
            json!({"input_cost_per_second": 0.02}),
        ),
        (
            "openai/requested".to_owned(),
            json!({"input_cost_per_second": 0.09}),
        ),
    ]));
    let events = [
        json!({"type": "transcription_session.created", "session": {"audio": {"input": {"transcription": {"model": "asr"}}}}}),
        json!({"type": "conversation.item.input_audio_transcription.completed", "usage": {"type": "duration", "seconds": 2.0}}),
        json!({"type": "conversation.item.input_audio_transcription.failed", "usage": {"type": "duration", "seconds": 100.0}}),
        json!({"type": "conversation.item.input_audio_transcription.completed", "usage": {"type": "duration", "seconds": 3.0}}),
    ];
    assert_eq!(
        catalog.handle_realtime_transcription_cost_calculation(&events, "openai", "requested"),
        5.0 * 0.02
    );
    assert_eq!(
        catalog.handle_realtime_transcription_cost_calculation(&events[..1], "openai", "requested"),
        0.0
    );
}
