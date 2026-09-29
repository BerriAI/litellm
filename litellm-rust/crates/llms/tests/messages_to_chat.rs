use litellm_llms::base_llm::messages::adapters::chat::prepare;
use litellm_llms_types::formats::{
    chat_completions::ChatCompletionsResponse, messages::MessagesRequest,
};
use rstest::rstest;
use serde_json::{Value, json};

fn request(value: Value) -> MessagesRequest {
    serde_json::from_value(value).expect("valid Messages request")
}

fn response(value: Value) -> ChatCompletionsResponse {
    serde_json::from_value(value).expect("valid Chat response")
}

#[rstest]
fn converts_a_tool_call_and_the_next_turn_result_without_changing_ids() {
    let first = prepare(
        request(json!({
            "model": "openai_like/model",
            "max_tokens": 32,
            "system": "Be brief",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"name": "weather", "description": "Get weather", "input_schema": {
                "type": "object", "properties": {"city": {"type": "string"}}
            }}]
        })),
        || "msg_fixed".into(),
    )
    .expect("first turn prepares");
    assert_eq!(first.messages[0].role, "system");
    assert_eq!(
        first.optional_params["tools"][0]["function"]["name"],
        "weather"
    );

    let answer = first.complete(response(json!({
        "created": 1,
        "model": "model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
        }]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0, "text_tokens": 0}}
    }))).expect("tool answer converts");
    assert_eq!(answer.id, "msg_fixed");
    assert_eq!(answer.stop_reason.as_deref(), Some("tool_use"));
    assert_eq!(answer.content[0]["id"], "call_1");
    assert_eq!(answer.content[0]["input"]["city"], "Paris");

    let second = prepare(request(json!({
        "model": "openai_like/model",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": answer.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}]}
        ]
    })), || "msg_second".into()).expect("second turn prepares");
    let wire = serde_json::to_value(second.messages).expect("serializable Chat messages");
    assert_eq!(wire[1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(wire[2]["tool_call_id"], "call_1");
    assert_eq!(wire[2]["content"], "sunny");
}

#[rstest]
#[case::reasoning(json!({"thinking": {"type": "enabled", "budget_tokens": 1000}}))]
#[case::continuation(json!({"container": {"id": "stateful"}}))]
#[case::unknown(json!({"unknown_option": true}))]
fn unsupported_options_decline_before_preparing(#[case] option: Value) {
    let body = Value::Object(
        json!({"model": "openai_like/model", "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}]})
        .as_object()
        .unwrap()
        .iter()
        .chain(option.as_object().unwrap())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect(),
    );
    let result = prepare(request(body), || {
        panic!("rejection must precede ID generation")
    });
    assert!(result.is_err());
}

#[rstest]
#[case::auto(json!({"type": "auto"}), json!("auto"))]
#[case::any(json!({"type": "any"}), json!("required"))]
#[case::named(json!({"type": "tool", "name": "weather"}),
    json!({"type": "function", "function": {"name": "weather"}}))]
fn tool_choice_keeps_the_callers_intent(#[case] choice: Value, #[case] expected: Value) {
    let prepared = prepare(
        request(json!({
            "model": "openai_like/model", "max_tokens": 32,
            "messages": [{"role": "user", "content": "weather?"}],
            "tool_choice": choice
        })),
        || "msg_1".into(),
    )
    .expect("tool choice translates");
    assert_eq!(prepared.optional_params["tool_choice"], expected);
}

#[rstest]
fn text_blocks_stay_in_one_user_turn() {
    let prepared = prepare(
        request(json!({
            "model": "openai_like/model", "max_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"}
            ]}]
        })),
        || "msg_1".into(),
    )
    .expect("text blocks translate");
    let wire = serde_json::to_value(prepared.messages).expect("Chat messages serialize");
    assert_eq!(
        wire,
        json!([{"role": "user", "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"}
        ]}])
    );
}
