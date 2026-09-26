#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_realtime_stream_combines_text_and_audio_token_details

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::realtime_cost::{
    collect_and_combine_usage_from_realtime_stream_results,
    collect_and_combine_usage_from_responses_ws_results, get_transcription_model_name_from_results,
    partition_results_by_service_tier, transcription_usage_cost,
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
        litellm_cost::cost_calculator::handle_realtime_transcription_cost_calculation(
            &catalog,
            &events,
            "openai",
            "requested"
        ),
        5.0 * 0.02
    );
    assert_eq!(
        litellm_cost::cost_calculator::handle_realtime_transcription_cost_calculation(
            &catalog,
            &events[..1],
            "openai",
            "requested"
        ),
        0.0
    );
}

#[rstest]
fn collect_realtime_usage_sums_billable_events_without_doubling_cache_writes() {
    let events = [
        json!({"type": "response.done", "response": {"usage": {"input_tokens": 100, "output_tokens": 10, "input_token_details": {"cached_tokens": 20, "cache_write_tokens": 5, "cached_tokens_details": {"audio_tokens": 12}}, "output_token_details": {"audio_tokens": 4}}}}),
        json!({"type": "response.delta", "response": {"usage": {"input_tokens": 1000, "output_tokens": 1000}}}),
        json!({"type": "response.done", "response": {"usage": {"input_tokens": 60, "output_tokens": 6, "input_token_details": {"cached_tokens": 10, "cache_write_tokens": 3, "cached_tokens_details": {"audio_tokens": 8}}, "output_token_details": {"audio_tokens": 2}}}}),
    ];
    let usage = collect_and_combine_usage_from_realtime_stream_results(&events).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (160, 16, 176)
    );
    assert_eq!(
        (
            prompt.cached_tokens,
            prompt.cache_write_tokens,
            prompt.cache_creation_tokens
        ),
        (30, Some(8), Some(8))
    );
    assert_eq!(prompt.cached_tokens_details.unwrap().audio_tokens, Some(20));
    assert_eq!(
        usage.completion_tokens_details.unwrap().audio_tokens,
        Some(6)
    );
}

#[rstest]
fn responses_ws_usage_filters_and_partitions_billable_events_by_service_tier() {
    let events = [
        json!({"type": "response.completed", "response": {"service_tier": "default", "usage": {"input_tokens": 100, "output_tokens": 40}}}),
        json!({"type": "response.failed", "response": {"service_tier": "priority", "usage": {"input_tokens": 1000, "output_tokens": 1000}}}),
        json!({"type": "response.incomplete", "response": {"service_tier": "priority", "usage": {"input_tokens": 60, "output_tokens": 10}}}),
        json!({"type": "response.completed", "response": {"service_tier": "priority", "usage": null}}),
        json!({"type": "response.completed", "response": {"service_tier": "default", "usage": {"input_tokens": 20, "output_tokens": 5}}}),
    ];
    let usage = collect_and_combine_usage_from_responses_ws_results(&events).unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (180, 55));
    let groups = partition_results_by_service_tier(&events);
    assert_eq!(
        groups
            .iter()
            .map(|(tier, group)| (*tier, group.len()))
            .collect::<Vec<_>>(),
        vec![(Some("default"), 2), (Some("priority"), 1)]
    );
}

#[rstest]
fn responses_ws_token_cost_prices_each_tier_at_its_returned_rate() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 0.002,
            "output_cost_per_token": 0.003,
            "input_cost_per_token_priority": 0.005,
            "output_cost_per_token_priority": 0.007
        }),
    )]));
    let events = [
        json!({"type": "response.completed", "response": {"service_tier": "default", "usage": {"input_tokens": 100, "output_tokens": 40}}}),
        json!({"type": "response.incomplete", "response": {"service_tier": "priority", "usage": {"input_tokens": 60, "output_tokens": 10}}}),
        json!({"type": "response.failed", "response": {"service_tier": "priority", "usage": {"input_tokens": 1000, "output_tokens": 1000}}}),
    ];
    let at: Timestamp = "2026-09-22T12:00:00Z".parse().unwrap();
    let actual = litellm_cost::cost_calculator::responses_ws_token_cost_by_tier(
        &catalog,
        &events,
        "model",
        Some("openai"),
        None,
        None,
        at,
    )
    .unwrap();
    let expected = 100.0 * 0.002 + 40.0 * 0.003 + 60.0 * 0.005 + 10.0 * 0.007;
    assert!((actual - expected).abs() < 1e-12);
    assert!((actual - (160.0 * 0.002 + 50.0 * 0.003)).abs() > 1e-9);
}

#[rstest]
#[case(json!({}), 100.0 * 0.002 + 20.0 * 0.003)]
#[case(json!({"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}), 0.0)]
fn handle_realtime_stream_cost_calculation_falls_through_only_for_priceless_session_models(
    #[case] session_info: Value,
    #[case] expected: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        ("openai/session".to_owned(), session_info),
        (
            "openai/requested".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
    ]));
    let events = [
        json!({"type": "session.created", "session": {"model": "session"}}),
        json!({"type": "response.done", "response": {"usage": {"input_tokens": 100, "output_tokens": 20}}}),
    ];
    let usage = collect_and_combine_usage_from_realtime_stream_results(&events).unwrap();
    let at: Timestamp = "2026-09-22T12:00:00Z".parse().unwrap();
    let actual = litellm_cost::cost_calculator::handle_realtime_stream_cost_calculation(
        &catalog,
        &events,
        &usage,
        "openai",
        "requested",
        None,
        at,
    );
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
fn handle_realtime_stream_cost_calculation_adds_transcription_events() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/session".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
        (
            "openai/asr".to_owned(),
            json!({"input_cost_per_second": 0.02}),
        ),
    ]));
    let events = [
        json!({"type": "session.created", "session": {"model": "session", "audio": {"input": {"transcription": {"model": "asr"}}}}}),
        json!({"type": "response.done", "response": {"usage": {"input_tokens": 100, "output_tokens": 20}}}),
        json!({"type": "conversation.item.input_audio_transcription.completed", "usage": {"type": "duration", "seconds": 2.0}}),
    ];
    let usage = collect_and_combine_usage_from_realtime_stream_results(&events).unwrap();
    let at: Timestamp = "2026-09-22T12:00:00Z".parse().unwrap();
    let actual = litellm_cost::cost_calculator::handle_realtime_stream_cost_calculation(
        &catalog,
        &events,
        &usage,
        "openai",
        "requested",
        None,
        at,
    );
    assert!((actual - (100.0 * 0.002 + 20.0 * 0.003 + 2.0 * 0.02)).abs() < 1e-12);
}

#[rstest]
#[case(true)]
#[case(false)]
fn realtime_combine_keeps_cached_split_when_only_one_usage_has_details(
    #[case] details_first: bool,
) {
    let with_details = json!({"type": "response.done", "response": {"usage": {
        "input_tokens": 283, "output_tokens": 0, "total_tokens": 283,
        "input_token_details": {
            "text_tokens": 116, "audio_tokens": 167, "cached_tokens": 192,
            "cached_tokens_details": {"text_tokens": 64, "audio_tokens": 128}
        }
    }}});
    let without_details = json!({"type": "response.done", "response": {"usage": {
        "input_tokens": 150, "output_tokens": 0, "total_tokens": 150,
        "input_token_details": {"text_tokens": 50, "audio_tokens": 100, "cached_tokens": 100}
    }}});
    let events = if details_first {
        [with_details, without_details]
    } else {
        [without_details, with_details]
    };
    let usage = collect_and_combine_usage_from_realtime_stream_results(&events).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 292);
    let split = prompt.cached_tokens_details.unwrap();
    assert_eq!(
        (split.text_tokens, split.audio_tokens),
        (Some(64), Some(128))
    );
}

#[rstest]
fn realtime_combine_leaves_cached_details_absent_when_no_event_carried_them() {
    let events = [
        json!({"type": "response.done", "response": {"usage": {
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
            "input_token_details": {"text_tokens": 100, "cached_tokens": 20}
        }}}),
        json!({"type": "response.done", "response": {"usage": {
            "input_tokens": 50, "output_tokens": 5, "total_tokens": 55,
            "input_token_details": {"text_tokens": 50, "cached_tokens": 10}
        }}}),
    ];
    let usage = collect_and_combine_usage_from_realtime_stream_results(&events).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 30);
    assert!(prompt.cached_tokens_details.is_none());
}
