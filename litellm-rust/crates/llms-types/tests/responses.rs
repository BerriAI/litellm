use litellm_llms_types::{
    formats::responses::streaming_websocket::{ResponsesWsEvent, ResponsesWsEventType},
    formats::responses::{
        ResponsesAnnotation, ResponsesApiResponse, ResponsesContentPart, ResponsesOutputItem,
    },
    recognized::Recognized,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::create("response.create", ResponsesWsEventType::ResponseCreate)]
#[case::created("response.created", ResponsesWsEventType::ResponseCreated)]
#[case::completed("response.completed", ResponsesWsEventType::ResponseCompleted)]
#[case::failed("response.failed", ResponsesWsEventType::ResponseFailed)]
#[case::incomplete("response.incomplete", ResponsesWsEventType::ResponseIncomplete)]
#[case::error("error", ResponsesWsEventType::Error)]
#[case::unknown(
    "response.output_text.delta",
    ResponsesWsEventType::Other("response.output_text.delta".to_string())
)]
#[case::empty("", ResponsesWsEventType::Other(String::new()))]
#[case::case_sensitive("Response.Completed", ResponsesWsEventType::Other("Response.Completed".into()))]
#[case::escaped("future\"\\\n", ResponsesWsEventType::Other("future\"\\\n".into()))]
fn websocket_event_type_round_trips(#[case] wire: &str, #[case] expected: ResponsesWsEventType) {
    let serialized = serde_json::to_string(&expected).unwrap();
    assert_eq!(serialized, serde_json::to_string(wire).unwrap());
    assert_eq!(
        serde_json::from_str::<ResponsesWsEventType>(&serialized).unwrap(),
        expected
    );
}

#[rstest]
#[case::number("17")]
#[case::boolean("true")]
#[case::null("null")]
#[case::array("[]")]
#[case::object("{}")]
fn websocket_event_type_rejects_non_strings(#[case] wire: &str) {
    assert!(serde_json::from_str::<ResponsesWsEventType>(wire).is_err());
}

#[cfg(feature = "schema")]
#[rstest]
fn websocket_event_type_schema_is_open_string() {
    let schema = schemars::schema_for!(ResponsesWsEventType);
    assert_eq!(schema.to_value().get("type"), Some(&json!("string")));
}

#[rstest]
fn response_output_exposes_content_annotations_and_function_arguments() {
    let wire = json!({"id":"resp_1","model":"model","output":[
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.test","start_index":0,"end_index":6}]}]},
        {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"q\"}"}
    ],"future":null});
    let response: ResponsesApiResponse = serde_json::from_value(wire.clone()).unwrap();
    let Some(ResponsesOutputItem::Message(message)) = response.output[0].known() else {
        panic!("expected message")
    };
    let Some(ResponsesContentPart::OutputText {
        text,
        annotations: Some(Recognized::Known(annotations)),
        ..
    }) = message.content.as_ref().unwrap().known().unwrap()[0].known()
    else {
        panic!("expected text")
    };
    assert_eq!(text, "answer");
    assert!(
        matches!(annotations[0].known(), Some(ResponsesAnnotation::UrlCitation(citation)) if citation.url == Some(Recognized::Known("https://example.test".into())))
    );
    assert!(
        matches!(response.output[1].known(), Some(ResponsesOutputItem::FunctionCall(call)) if call.name == Some(Recognized::Known("lookup".into())) && call.arguments == Some(Recognized::Known("{\"query\":\"q\"}".into())))
    );
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::reasoning(json!({"type":"reasoning","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":null}))]
#[case::web_search(json!({"type":"web_search_call","action":{"type":"search","queries":["q"],"sources":[{"type":"url","url":"https://example.test"}]}}))]
#[case::file_search(json!({"type":"file_search_call","queries":["q"],"results":[{"file_id":"file_1","score":1,"attributes":{"custom":[1,null]}}]}))]
#[case::code(json!({"type":"code_interpreter_call","outputs":[{"type":"logs","logs":"done"},{"type":"image","url":"https://example.test"}]}))]
#[case::image(json!({"type":"image_generation_call","result":null}))]
#[case::custom(json!({"type":"custom_tool_call","name":"lookup","input":"q"}))]
#[case::mcp(json!({"type":"mcp_call","server_label":"server","name":"lookup","arguments":"{}","output":null}))]
fn known_output_items_round_trip(#[case] wire: Value) {
    let parsed: Recognized<ResponsesOutputItem> = serde_json::from_value(wire.clone()).unwrap();
    assert!(parsed.known().is_some());
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::scalar(json!(17))]
#[case::null(json!(null))]
#[case::unknown(json!({"type":"future","payload":[1,null]}))]
#[case::partial_known(json!({"type":"message","content":[{"type":"output_text","text":17},null],"role":false}))]
fn formerly_opaque_output_remains_lossless(#[case] output: Value) {
    let wire = json!({"id":"resp_1","model":"model","output":[output]});
    let parsed: ResponsesApiResponse = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::flat(json!({"type":"response.create","model":"flat","response":{"model":"nested"}}), Some("flat"))]
#[case::wrong_flat(json!({"type":"response.created","model":17,"response":{"model":"nested","output":null}}), Some("nested"))]
#[case::null_response(json!({"type":"future","model":null,"response":null,"vendor":true}), None)]
#[case::partial(json!({"type":"response.created","response":{"model":false,"status":null,"future":[1,null]}}), None)]
fn websocket_metadata_preserves_permissive_shapes_and_model_precedence(
    #[case] wire: Value,
    #[case] model: Option<&str>,
) {
    let event: ResponsesWsEvent = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(event.model(), model);
    assert_eq!(serde_json::to_value(event).unwrap(), wire);
}
