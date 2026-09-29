use litellm_llms_types::{
    formats::responses::{
        ResponsesApiResponse, ResponsesContent, ResponsesContentPart, ResponsesInput,
        ResponsesItem, ResponsesRequest,
    },
    recognized::Recognized,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
fn response_exposes_message_reasoning_and_calls_without_losing_extensions() {
    let wire = json!({
        "id": "response-1", "model": "test-model", "status": "completed",
        "output": [
            {"type": "reasoning", "id": "reasoning-1", "summary": [], "content": [{"type": "reasoning_text", "text": "thinking", "future": null}]},
            {"type": "message", "id": "message-1", "role": "assistant", "content": [{"type": "output_text", "text": "hello", "annotations": [], "future": null}]},
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{\"key\":null}", "status": null},
            {"type": "future_item", "payload": {"nested": [null, 1]}}
        ],
        "usage": {"input_tokens": 3, "provider_counter": null},
        "provider_field": [null, true]
    });
    let response: ResponsesApiResponse = serde_json::from_value(wire.clone()).unwrap();
    assert!(
        matches!(response.output[0].known(), Some(ResponsesItem::Reasoning { summary, .. }) if summary.is_empty())
    );
    let Some(ResponsesItem::Message {
        content: ResponsesContent::Parts(parts),
        ..
    }) = response.output[1].known()
    else {
        panic!("expected typed message content");
    };
    assert!(
        matches!(parts[0].known(), Some(ResponsesContentPart::OutputText { text, .. }) if text == "hello")
    );
    assert!(
        matches!(response.output[2].known(), Some(ResponsesItem::FunctionCall { call_id, name, arguments, .. }) if call_id == "call-1" && name == "lookup" && arguments == "{\"key\":null}")
    );
    assert!(matches!(response.output[3], Recognized::Unrecognized(_)));
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::message(json!({"type": "message", "role": "user", "content": "hi", "id": null}), true)]
#[case::message_parts(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi", "future": null}, {"type": "future_part", "data": null}]}), true)]
#[case::function_call(json!({"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "incomplete {"}), true)]
#[case::function_output_text(json!({"type": "function_call_output", "call_id": "call-1", "output": "ok", "status": null}), true)]
#[case::function_output_parts(json!({"type": "function_call_output", "call_id": "call-1", "output": [{"type": "input_image", "image_url": "https://example.test/image", "detail": null}]}), true)]
#[case::reasoning(json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary", "extra": null}], "encrypted_content": null}), true)]
#[case::shorthand_message(json!({"role": "user", "content": "hi"}), false)]
#[case::unknown(json!({"type": "future_item", "role": "assistant", "content": "hi"}), false)]
#[case::null_content(json!({"type": "message", "role": "assistant", "content": null}), false)]
#[case::missing_call_id(json!({"type": "function_call", "name": "lookup", "arguments": "{}"}), false)]
#[case::wrong_summary_shape(json!({"type": "reasoning", "summary": "summary"}), false)]
#[case::null(json!(null), false)]
fn items_recognize_supported_shapes_and_preserve_every_value(
    #[case] wire: Value,
    #[case] known: bool,
) {
    let item: Recognized<ResponsesItem> = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(item.known().is_some(), known);
    assert_eq!(serde_json::to_value(item).unwrap(), wire);
}

#[rstest]
#[case::input_text(json!({"type": "input_text", "text": "hi", "extra": null}), true)]
#[case::output_text(json!({"type": "output_text", "text": "hi", "annotations": []}), true)]
#[case::image(json!({"type": "input_image", "image_url": "https://example.test/image", "detail": "auto"}), true)]
#[case::summary(json!({"type": "summary_text", "text": "summary"}), true)]
#[case::reasoning(json!({"type": "reasoning_text", "text": "reasoning"}), true)]
#[case::unknown(json!({"type": "future_content", "text": "hi"}), false)]
#[case::wrong_text_type(json!({"type": "output_text", "text": null}), false)]
#[case::image_by_file(json!({"type": "input_image", "file_id": "file-1"}), false)]
fn content_parts_keep_discriminators_and_nested_extensions(
    #[case] wire: Value,
    #[case] known: bool,
) {
    let part: Recognized<ResponsesContentPart> = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(part.known().is_some(), known);
    assert_eq!(serde_json::to_value(part).unwrap(), wire);
}

#[rstest]
#[case::text(json!("hi"))]
#[case::items(json!([
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
    {"role": "user", "content": "shorthand"},
    {"type": "future_item", "value": null},
    null
]))]
fn request_preserves_input_and_unmodeled_parameters(#[case] input: Value) {
    let wire = json!({"model": "test-model", "input": input,
        "reasoning": {"effort": "future-effort", "extension": null},
        "previous_response_id": null, "store": false, "provider_field": [1, null]});
    let request: ResponsesRequest = serde_json::from_value(wire.clone()).unwrap();
    match &request.input {
        ResponsesInput::Text(text) => assert_eq!(text, "hi"),
        ResponsesInput::Items(items) => assert!(matches!(
            items[0].known(),
            Some(ResponsesItem::Message { .. })
        )),
    }
    assert_eq!(serde_json::to_value(request).unwrap(), wire);
}

#[rstest]
#[case::null(json!(null))]
#[case::number(json!(7))]
#[case::object(json!({"role": "user", "content": "hi"}))]
fn input_rejects_shapes_outside_the_existing_request_contract(#[case] input: Value) {
    assert!(serde_json::from_value::<ResponsesInput>(input).is_err());
}
