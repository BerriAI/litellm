use litellm_types::messages::streaming::MessagesStreamEvent;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::start(json!({
    "type": "message_start",
    "message": {
        "id": "msg", "type": "message", "role": "assistant", "model": "test-model",
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "future_usage": {"count": 9}},
        "future_message": [1, 2]
    }
}))]
#[case::block(json!({
    "type": "content_block_start", "index": 0,
    "content_block": {"type": "future", "payload": {"keep": true}}
}))]
#[case::delta(json!({
    "type": "message_delta", "delta": {"stop_reason": "end_turn", "future_delta": 42},
    "usage": {"output_tokens": 7, "future_usage": true}
}))]
#[case::error(json!({
    "type": "error", "error": {"type": "future_error", "message": "failed", "future": 42}
}))]
fn stream_events_preserve_extensible_fields(#[case] wire: Value) {
    let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(event).unwrap(), wire);
}

#[rstest]
#[case::invalid_index(json!({"type": "content_block_stop", "index": "zero"}))]
#[case::missing_delta(json!({"type": "content_block_delta", "index": 0}))]
#[case::unknown_event(json!({"type": "future_event"}))]
fn malformed_or_unrecognized_events_remain_rejected(#[case] wire: Value) {
    assert!(serde_json::from_value::<MessagesStreamEvent>(wire).is_err());
}
