use litellm_llms::base_llm::messages::normalization::normalize_system_role_messages;
use litellm_llms_types::formats::messages::MessagesRequest;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text_system(json!("existing"), json!([{"type": "text", "text": "existing"}]))]
#[case::block_system(json!([{ "type": "future", "payload": 7 }]), json!([{ "type": "future", "payload": 7 }]))]
#[case::no_system(Value::Null, json!([]))]
fn hoisting_preserves_block_fields_order_and_unrelated_request_fields(
    #[case] system: Value,
    #[case] initial_blocks: Value,
) {
    let cache_control = json!({"type": "ephemeral", "scope": "global", "future": true});
    let leading_block = json!({"type": "text", "text": "second", "cache_control": cache_control});
    let user = json!({"role": "user", "content": "hello", "future_message": 42});
    let later_turn = json!({"role": "system", "content": "mid-conversation reminder"});
    let request: MessagesRequest = serde_json::from_value(json!({
        "model": "test-model",
        "max_tokens": 64,
        "system": system,
        "messages": [
            {"role": "system", "content": "first"},
            {"role": "system", "content": [leading_block]},
            user,
            later_turn
        ],
        "future_request": {"nested": true}
    }))
    .unwrap();
    let normalized = normalize_system_role_messages(request, true);
    let expected_blocks: Vec<Value> = initial_blocks
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .chain([json!({"type": "text", "text": "first"}), leading_block])
        .collect();
    assert_eq!(
        serde_json::to_value(&normalized).unwrap(),
        json!({
            "model": "test-model",
            "max_tokens": 64,
            "system": expected_blocks,
            "messages": [user, later_turn],
            "future_request": {"nested": true}
        })
    );
    assert_eq!(
        normalize_system_role_messages(normalized.clone(), true),
        normalized
    );
}
