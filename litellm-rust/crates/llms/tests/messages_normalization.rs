use litellm_llms::base_llm::messages::normalization::fold_system_role_messages;
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text_system(json!("existing"), json!([{"type": "text", "text": "existing"}]))]
#[case::block_system(json!([{ "type": "future", "payload": 7 }]), json!([{ "type": "future", "payload": 7 }]))]
#[case::no_system(Value::Null, json!([]))]
fn folding_preserves_block_fields_order_and_unrelated_request_fields(
    #[case] system: Value,
    #[case] initial_blocks: Value,
) {
    let cache_control = json!({"type": "ephemeral", "scope": "global", "future": true});
    let folded_block = json!({"type": "text", "text": "second", "cache_control": cache_control});
    let user = json!({"role": "user", "content": "hello", "future_message": 42});
    let request: AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "test-model",
        "max_tokens": 64,
        "system": system,
        "messages": [
            {"role": "system", "content": "first"},
            user,
            {"role": "system", "content": [folded_block]}
        ],
        "future_request": {"nested": true}
    }))
    .unwrap();
    let folded = fold_system_role_messages(request);
    let expected_blocks: Vec<Value> = initial_blocks
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .chain([json!({"type": "text", "text": "first"}), folded_block])
        .collect();
    assert_eq!(
        serde_json::to_value(&folded).unwrap(),
        json!({
            "model": "test-model",
            "max_tokens": 64,
            "system": expected_blocks,
            "messages": [user],
            "future_request": {"nested": true}
        })
    );
    assert_eq!(fold_system_role_messages(folded.clone()), folded);
}
