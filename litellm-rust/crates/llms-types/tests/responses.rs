use litellm_llms_types::formats::responses::{
    ResponsesOutputItem,
    streaming_websocket::{ResponsesEventResponse, ResponsesWsEventType},
};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T>(wire: Value)
where
    T: DeserializeOwned + Serialize,
{
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

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
#[case::case_sensitive(
    "Response.Completed",
    ResponsesWsEventType::Other("Response.Completed".into())
)]
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
    assert_eq!(
        schema.to_value().get("type"),
        Some(&serde_json::json!("string"))
    );
}

#[rstest]
#[case::message(json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.test","start_index":0,"end_index":6}]}]}))]
#[case::function_call(json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"q\"}"}))]
#[case::custom_tool(json!({"type":"custom_tool_call","name":"lookup","input":"q"}))]
#[case::reasoning(json!({"type":"reasoning","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"opaque"}))]
#[case::web_search(json!({"type":"web_search_call","action":{"type":"search","queries":["q"],"sources":[{"type":"url","url":"https://example.test"}]}}))]
#[case::file_search(json!({"type":"file_search_call","queries":["q"],"results":[{"file_id":"file_1","score":1,"attributes":{"custom":[1,null]}}]}))]
#[case::code(json!({"type":"code_interpreter_call","outputs":[{"type":"logs","logs":"done"},{"type":"image","url":"https://example.test"}]}))]
#[case::image(json!({"type":"image_generation_call","result":"generated"}))]
#[case::mcp(json!({"type":"mcp_call","server_label":"server","name":"lookup","arguments":"{}","output":"done"}))]
fn output_items_round_trip(#[case] wire: Value) {
    round_trip::<ResponsesOutputItem>(wire);
}

#[rstest]
fn nested_event_response_exposes_typed_output_and_preserves_extensions() {
    let wire = json!({
        "id":"response_1",
        "model":"example-model",
        "status":"completed",
        "output":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","extension":true}],
        "extension":{"nested":[1,null]}
    });
    let response: ResponsesEventResponse = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(response.id.as_deref(), Some("response_1"));
    assert_eq!(response.model.as_deref(), Some("example-model"));
    let Some(output) = &response.output else {
        panic!("expected typed output");
    };
    let [ResponsesOutputItem::FunctionCall(call)] = output.as_slice() else {
        panic!("expected function call");
    };
    assert_eq!(call.name.as_deref(), Some("lookup"));
    assert_eq!(call.arguments.as_deref(), Some("{}"));
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::empty(json!({}))]
#[case::partial(json!({"id":"response_1","output":[]}))]
fn nested_event_response_accepts_partial_metadata(#[case] wire: Value) {
    round_trip::<ResponsesEventResponse>(wire);
}
