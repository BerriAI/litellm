use litellm_llms_types::formats::chat_completions::ChatMessage;
use rstest::rstest;
use serde_json::json;

#[rstest]
fn wire_type_preserves_serialization() {
    let message = ChatMessage {
        role: "user".to_owned(),
        content: None,
        name: None,
        extra: Default::default(),
    };

    assert_eq!(
        serde_json::to_value(message).unwrap(),
        json!({"role": "user"})
    );
}

#[cfg(feature = "schema")]
#[rstest]
fn wire_type_supports_schema_generation() {
    let schema = schemars::schema_for!(ChatMessage);

    assert!(
        schema
            .to_value()
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|properties| properties.contains_key("role"))
    );
}
