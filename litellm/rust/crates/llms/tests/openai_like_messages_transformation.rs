use litellm_llms::openai_like::messages::transformation::{
    complete_messages_url, portable_cache_control,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::root("https://native.test", "https://native.test/v1/messages")]
#[case::trailing_slash("https://native.test/", "https://native.test/v1/messages")]
#[case::versioned("https://native.test/v1/", "https://native.test/v1/messages")]
#[case::full_url(
    "https://native.test/v3/v1/messages/",
    "https://native.test/v3/v1/messages"
)]
#[case::gateway_prefix("https://native.test/v3", "https://native.test/v3/v1/messages")]
fn messages_url_appends_the_endpoint_once(#[case] base: &str, #[case] expected: &str) {
    assert_eq!(complete_messages_url(base), expected);
}

#[rstest]
#[case::extended(json!({"type": "ephemeral", "ttl": "1h", "scope": "global"}), Some(json!({"type": "ephemeral"})))]
#[case::missing_type(json!({"ttl": "1h"}), Some(json!({"type": "ephemeral"})))]
#[case::invalid_type(json!({"type": 7}), Some(json!({"type": "ephemeral"})))]
#[case::null(Value::Null, None)]
#[case::scalar(json!(false), None)]
fn portable_cache_only_rewrites_protocol_cache_hints(
    #[case] cache: Value,
    #[case] expected_cache: Option<Value>,
) {
    let application_data = json!({"cache_control": cache, "nested": {"cache_control": cache}});
    let body = json!({
        "cache_control": cache,
        "model": "native-test",
        "system": [{"type": "text", "text": "system", "cache_control": cache}],
        "tools": [{"name": "tool", "cache_control": cache, "input_schema": application_data}],
        "messages": [{"role": "user", "content": [
            {"type": "tool_use", "input": application_data, "cache_control": cache},
            {"type": "tool_result", "content": [{"type": "text", "text": "result", "cache_control": cache}], "cache_control": cache},
        ]}],
    });
    let hint = |value: Value| {
        let fields = value.as_object().unwrap().clone();
        Value::Object(
            fields
                .into_iter()
                .chain(
                    expected_cache
                        .clone()
                        .map(|cache| ("cache_control".into(), cache)),
                )
                .collect(),
        )
    };
    let expected = hint(json!({
        "model": "native-test",
        "system": [hint(json!({"type": "text", "text": "system"}))],
        "tools": [hint(json!({"name": "tool", "input_schema": application_data}))],
        "messages": [{"role": "user", "content": [
            hint(json!({"type": "tool_use", "input": application_data})),
            hint(json!({"type": "tool_result", "content": [hint(json!({"type": "text", "text": "result"}))]})),
        ]}],
    }));
    assert_eq!(portable_cache_control(body), expected);
}
