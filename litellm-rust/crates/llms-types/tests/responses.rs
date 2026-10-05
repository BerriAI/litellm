use litellm_llms_types::formats::responses::streaming_websocket::ResponsesWsEventType;
use rstest::rstest;

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
