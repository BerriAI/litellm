use litellm_llms_types::serde_compat::Nullable;
use litellm_llms_types::{
    formats::messages::{
        Citation, Citations, ContentBlockType, MessageType, MessagesResponse, StopDetailsType,
        StopReason, UsageIterationType,
    },
    recognized::Recognized,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[rstest]
#[case::known("message", MessageType::Message)]
#[case::unknown("future_type", MessageType::Other("future_type".into()))]
#[case::case_sensitive("Message", MessageType::Other("Message".into()))]
#[case::empty("", MessageType::Other(String::new()))]
fn message_type_round_trips_as_a_string(#[case] wire: &str, #[case] expected: MessageType) {
    let parsed: MessageType = serde_json::from_value(json!(wire)).unwrap();
    assert_eq!(parsed, expected);
    assert_eq!(serde_json::to_value(parsed).unwrap(), json!(wire));
}

#[rstest]
#[case::null(json!(null))]
#[case::number(json!(1))]
#[case::boolean(json!(true))]
#[case::array(json!(["message"]))]
#[case::object(json!({"type": "message"}))]
fn message_type_rejects_non_string_json(#[case] value: Value) {
    assert!(serde_json::from_value::<MessageType>(value).is_err());
}

#[cfg(feature = "schema")]
#[rstest]
fn message_type_schema_remains_named_and_string_compatible() {
    let schema = schemars::schema_for!(MessageType).to_value();
    assert_eq!(schema.get("title"), Some(&json!(stringify!(MessageType))));
    assert_eq!(schema.get("type"), Some(&json!("string")));
}

#[rstest]
#[case::end_turn("end_turn", StopReason::EndTurn)]
#[case::max_tokens("max_tokens", StopReason::MaxTokens)]
#[case::stop_sequence("stop_sequence", StopReason::StopSequence)]
#[case::tool_use("tool_use", StopReason::ToolUse)]
#[case::refusal("refusal", StopReason::Refusal)]
#[case::compaction("compaction", StopReason::Compaction)]
#[case::pause_turn("pause_turn", StopReason::PauseTurn)]
#[case::model_context_window_exceeded(
    "model_context_window_exceeded",
    StopReason::ModelContextWindowExceeded
)]
#[case::unknown("future_reason", StopReason::Other("future_reason".into()))]
#[case::case_sensitive("Refusal", StopReason::Other("Refusal".into()))]
#[case::empty("", StopReason::Other(String::new()))]
fn stop_reason_round_trips_as_a_string(#[case] wire: &str, #[case] expected: StopReason) {
    let parsed: StopReason = serde_json::from_value(json!(wire)).unwrap();
    assert_eq!(parsed, expected);
    assert_eq!(serde_json::to_value(parsed).unwrap(), json!(wire));
}

#[rstest]
#[case::null(json!(null))]
#[case::number(json!(1))]
#[case::boolean(json!(true))]
#[case::array(json!(["refusal"]))]
#[case::object(json!({"reason": "refusal"}))]
fn stop_reason_rejects_non_string_json(#[case] value: Value) {
    assert!(serde_json::from_value::<StopReason>(value).is_err());
}

#[cfg(feature = "schema")]
#[rstest]
fn stop_reason_schema_remains_named_and_string_compatible() {
    let schema = schemars::schema_for!(StopReason).to_value();
    assert_eq!(schema.get("title"), Some(&json!(stringify!(StopReason))));
    assert_eq!(schema.get("type"), Some(&json!("string")));
}

#[fixture]
fn response_wire() -> Value {
    json!({
        "id":"msg_1","type":"message","role":"assistant","model":"test-model",
        "content":[
            {"type":"text","text":"answer","citations":[{"type":"char_location","cited_text":"doc","document_index":0,"document_title":null,"start_char_index":0,"end_char_index":3}]},
            {"type":"compaction","content":null,"signature":"signed"}
        ],
        "stop_reason":"refusal","stop_sequence":null,
        "stop_details":{"type":"refusal","category":null,"explanation":"reason"},
        "usage":{"input_tokens":9,"output_tokens":3,"server_tool_use":{"web_search_requests":2},"iterations":[{"type":"compaction","input_tokens":7,"output_tokens":1},{"type":"message","input_tokens":2,"output_tokens":2}],"future":null},
        "context_management":{"applied_edits":[{"type":"compact_20260112","summary_input_tokens":7,"summary_output_tokens":1,"warnings":["warning"]}]},
        "safeguard_results":[{"future":[1,null]}],"future_response":true
    })
}

#[rstest]
fn response_exposes_content_usage_and_metadata(response_wire: Value) {
    let response: MessagesResponse = serde_json::from_value(response_wire.clone()).unwrap();
    assert_eq!(response.stop_reason, Some(StopReason::Refusal));
    assert_eq!(
        response
            .stop_details
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .detail_type,
        StopDetailsType::Refusal
    );
    let text = response.content[0].known().unwrap();
    assert_eq!(
        text.block_type,
        Some(Nullable::Value(ContentBlockType::Text))
    );
    let Some(Recognized::Known(Citations::Results(citations))) = &text.citations else {
        panic!("expected citations")
    };
    assert!(
        matches!(citations[0].known(), Some(Citation::CharLocation(citation)) if citation.start_char_index == Some(Recognized::Known(0)) && citation.end_char_index == Some(Recognized::Known(3)))
    );
    let usage = response.usage.as_ref().unwrap().known().unwrap();
    assert_eq!(usage.input_tokens, Some(Nullable::Value(9)));
    assert_eq!(
        usage
            .server_tool_use
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .web_search_requests,
        Some(Recognized::Known(2))
    );
    let iterations = usage.iterations.as_ref().unwrap().known().unwrap();
    assert_eq!(
        iterations[0].known().unwrap().iteration_type,
        UsageIterationType::Compaction
    );
    assert_eq!(
        iterations[1].known().unwrap().iteration_type,
        UsageIterationType::Message
    );
    let edits = response
        .context_management
        .as_ref()
        .unwrap()
        .known()
        .unwrap()
        .applied_edits
        .as_ref()
        .unwrap()
        .known()
        .unwrap();
    assert_eq!(
        edits[0].known().unwrap().summary_input_tokens,
        Some(Recognized::Known(7))
    );
    assert_eq!(serde_json::to_value(response).unwrap(), response_wire);
}

#[rstest]
#[case::null(json!(null))]
#[case::scalar(json!(17))]
#[case::unknown(json!({"type":"future","payload":[1,null]}))]
#[case::wrong_known_shape(json!({"type":"text","text":17}))]
#[case::nullable_known_fields(json!({"type":"thinking","thinking":null,"signature":null}))]
fn previously_opaque_response_content_remains_accepted(
    response_wire: Value,
    #[case] content: Value,
) {
    let wire = replace_field(response_wire, "content", json!([content]));
    let response: MessagesResponse = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::null(json!(null))]
#[case::scalar(json!(17))]
#[case::wrong_known_shape(json!({"input_tokens":"many","iterations":"future"}))]
#[case::nullable_counts(json!({"input_tokens":null,"output_tokens":null}))]
fn previously_opaque_usage_remains_accepted(response_wire: Value, #[case] usage: Value) {
    let wire = replace_field(response_wire, "usage", usage);
    let response: MessagesResponse = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

fn replace_field(wire: Value, field: &str, replacement: Value) -> Value {
    Value::Object(
        wire.as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.as_str() != field)
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([(field.to_string(), replacement)])
            .collect(),
    )
}

#[rstest]
fn nested_server_results_and_extended_usage_are_accessible(response_wire: Value) {
    use litellm_llms_types::formats::messages::BlockContent;
    let content = json!([{"type":"code_execution_tool_result","tool_use_id":"tool_1","content":{"type":"code_execution_result","stdout":"done","stderr":"","return_code":0,"content":[{"type":"code_execution_output","file_id":"file_1"}]}}]);
    let usage = json!({"input_tokens":1,"output_tokens":2,"cache_creation":{"ephemeral_1h_input_tokens":3,"ephemeral_5m_input_tokens":4},"server_tool_use":{"web_fetch_requests":5},"output_tokens_details":{"thinking_tokens":1},"speed":"fast","service_tier":"priority"});
    let wire = replace_field(
        replace_field(response_wire, "content", content),
        "usage",
        usage,
    );
    let parsed: MessagesResponse = serde_json::from_value(wire.clone()).unwrap();
    let Some(Recognized::Known(BlockContent::Block(result))) =
        &parsed.content[0].known().unwrap().content
    else {
        panic!("expected result block")
    };
    assert_eq!(result.stdout, Some(Recognized::Known("done".into())));
    assert_eq!(result.return_code, Some(Recognized::Known(0)));
    let Some(Recognized::Known(BlockContent::Blocks(outputs))) = &result.content else {
        panic!("expected outputs")
    };
    assert_eq!(
        outputs[0].known().unwrap().file_id,
        Some(Recognized::Known("file_1".into()))
    );
    let usage = parsed.usage.as_ref().unwrap().known().unwrap();
    assert_eq!(
        usage
            .cache_creation
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .ephemeral_1h_input_tokens,
        Some(Recognized::Known(3))
    );
    assert_eq!(
        usage
            .server_tool_use
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .web_fetch_requests,
        Some(Recognized::Known(5))
    );
    assert_eq!(
        usage
            .output_tokens_details
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .thinking_tokens,
        Some(Recognized::Known(1))
    );
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}
