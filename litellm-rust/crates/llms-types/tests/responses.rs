use litellm_llms_types::formats::responses::{
    ResponsesCodeOutput, ResponsesContentPart, ResponsesMcpError, ResponsesMcpErrorDetail,
    ResponsesOutputItem, ResponsesWebSearchAction,
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
    let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
    match &item {
        ResponsesOutputItem::Message(message) => {
            assert_eq!(message.role.as_deref(), Some("assistant"));
            let Some(content) = &message.content else {
                panic!("expected content")
            };
            let [
                ResponsesContentPart::OutputText {
                    text,
                    annotations: Some(annotations),
                    ..
                },
            ] = content.as_slice()
            else {
                panic!("expected output text and annotations")
            };
            assert_eq!(text, "answer");
            let [
                litellm_llms_types::formats::responses::ResponsesAnnotation::UrlCitation(citation),
            ] = annotations.as_slice()
            else {
                panic!("expected URL citation")
            };
            assert_eq!(citation.url.as_deref(), Some("https://example.test"));
            assert_eq!(citation.start_index, Some(0));
        }
        ResponsesOutputItem::FunctionCall(call) => {
            assert_eq!(call.call_id.as_deref(), Some("call_1"));
            assert_eq!(call.name.as_deref(), Some("lookup"));
            assert_eq!(call.arguments.as_deref(), Some("{\"query\":\"q\"}"));
        }
        ResponsesOutputItem::CustomToolCall(call) => {
            assert_eq!(call.name.as_deref(), Some("lookup"));
            assert_eq!(call.input.as_deref(), Some("q"));
        }
        ResponsesOutputItem::Reasoning(reasoning) => {
            assert_eq!(reasoning.encrypted_content.as_deref(), Some("opaque"));
            let Some(summary) = &reasoning.summary else {
                panic!("expected summary")
            };
            let [ResponsesContentPart::SummaryText { text, .. }] = summary.as_slice() else {
                panic!("expected summary text")
            };
            assert_eq!(text, "summary");
        }
        ResponsesOutputItem::WebSearchCall(call) => {
            let Some(ResponsesWebSearchAction::Search {
                queries: Some(queries),
                sources: Some(sources),
                ..
            }) = &call.action
            else {
                panic!("expected search action")
            };
            assert_eq!(queries, &["q"]);
            assert_eq!(sources[0].url.as_deref(), Some("https://example.test"));
        }
        ResponsesOutputItem::FileSearchCall(call) => {
            assert_eq!(call.queries.as_deref(), Some(["q".to_owned()].as_slice()));
            let Some(results) = &call.results else {
                panic!("expected search results")
            };
            assert_eq!(results[0].file_id.as_deref(), Some("file_1"));
            assert_eq!(results[0].score, Some(1.into()));
            assert_eq!(
                results[0].attributes.as_ref().unwrap()["custom"],
                json!([1, null])
            );
        }
        ResponsesOutputItem::CodeInterpreterCall(call) => {
            let Some(outputs) = &call.outputs else {
                panic!("expected code outputs")
            };
            let [
                ResponsesCodeOutput::Logs { logs, .. },
                ResponsesCodeOutput::Image { url, .. },
            ] = outputs.as_slice()
            else {
                panic!("expected logs and image")
            };
            assert_eq!(logs, "done");
            assert_eq!(url, "https://example.test");
        }
        ResponsesOutputItem::ImageGenerationCall(call) => {
            assert_eq!(call.result.as_deref(), Some("generated"))
        }
        ResponsesOutputItem::McpListTools(_) => panic!("discovery has a separate payload case"),
        ResponsesOutputItem::McpCall(call) => {
            assert_eq!(call.server_label.as_deref(), Some("server"));
            assert_eq!(call.name.as_deref(), Some("lookup"));
            assert_eq!(call.arguments.as_deref(), Some("{}"));
            assert_eq!(call.output.as_deref(), Some("done"));
        }
    }
    assert_eq!(serde_json::to_value(item).unwrap(), wire);
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

#[rstest]
#[case::missing_tag(json!({"output":[{}]}))]
#[case::unknown_tag(json!({"output":[{"type":"future"}]}))]
#[case::wrong_model(json!({"model":7}))]
#[case::wrong_output(json!({"output":{}}))]
#[case::wrong_known_nested_field(json!({"output":[{"type":"message","content":[{"type":"output_text","text":7}]}]}))]
#[case::wrong_action(json!({"output":[{"type":"web_search_call","action":{"type":"find","url":"u","pattern":false}}]}))]
fn event_response_rejects_malformed_typed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ResponsesEventResponse>(wire).is_err());
}

#[rstest]
fn event_response_optional_fields_omit_missing_and_null() {
    let response: ResponsesEventResponse = serde_json::from_value(json!({
        "id":null,"model":null,"status":null,"output":null,"future":null
    }))
    .unwrap();
    assert!(response.id.is_none());
    assert!(response.model.is_none());
    assert!(response.status.is_none());
    assert!(response.output.is_none());
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({"future":null})
    );
}

#[rstest]
#[case::refusal(json!({"type":"refusal","refusal":"refused","future":null}), ResponsesContentPart::Refusal { refusal:"refused".into(), extra:serde_json::Map::from_iter([("future".into(), Value::Null)]) })]
#[case::reasoning(json!({"type":"reasoning_text","text":"reason"}), ResponsesContentPart::ReasoningText { text:"reason".into(), extra:Default::default() })]
fn content_parts_expose_typed_variants(
    #[case] wire: Value,
    #[case] expected: ResponsesContentPart,
) {
    let content: ResponsesContentPart = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(content, expected);
    assert_eq!(serde_json::to_value(content).unwrap(), wire);
}

#[rstest]
fn mcp_discovery_exposes_tools_and_nested_schemas() {
    let wire = json!({
        "type":"mcp_list_tools","id":"item_1","server_label":"tools",
        "tools":[{"name":"lookup","description":"Look up a value","input_schema":{"type":"object","properties":{"query":{"type":"string"}}},"annotations":{"read_only":false},"future":null}],
        "extension":[1,null]
    });
    let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
    let ResponsesOutputItem::McpListTools(discovery) = &item else {
        panic!("expected discovery")
    };
    assert_eq!(discovery.server_label.as_deref(), Some("tools"));
    let tool = &discovery.tools.as_ref().unwrap()[0];
    assert_eq!(tool.name.as_deref(), Some("lookup"));
    let Some(litellm_llms_types::json_schema::JsonSchema::Object(schema)) = &tool.input_schema
    else {
        panic!("expected tool schema")
    };
    assert!(schema.properties.as_ref().unwrap().contains_key("query"));
    assert_eq!(tool.annotations, Some(json!({"read_only":false})));
    assert_eq!(tool.extra["future"], Value::Null);
    assert_eq!(serde_json::to_value(item).unwrap(), wire);
}

#[rstest]
#[case::partial(json!({"server_label":"tools","tools":[]}))]
#[case::failure(json!({"server_label":"tools","error":"unavailable"}))]
fn mcp_discovery_accepts_partial_payloads(#[case] wire: Value) {
    let discovery: litellm_llms_types::formats::responses::ResponsesMcpListTools =
        serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(discovery.server_label.as_deref(), Some("tools"));
    assert!(discovery.id.is_none());
    assert_eq!(serde_json::to_value(discovery).unwrap(), wire);
}

#[rstest]
#[case::message(json!("unavailable"), ResponsesMcpError::Message("unavailable".into()))]
#[case::protocol(json!({"type":"mcp_protocol_error","code":-32600,"message":"invalid","extension":null}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::McpProtocolError {code:-32600,message:"invalid".into(),extra:serde_json::Map::from_iter([("extension".into(), Value::Null)])}))]
#[case::http(json!({"type":"http_error","code":503,"message":"unavailable"}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::HttpError {code:503,message:"unavailable".into(),extra:Default::default()}))]
#[case::tool(json!({"type":"mcp_tool_execution_error","content":{"arbitrary":[1,null]}}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::McpToolExecutionError {content:json!({"arbitrary":[1,null]}),extra:Default::default()}))]
fn mcp_call_errors_expose_typed_variants(#[case] wire: Value, #[case] expected: ResponsesMcpError) {
    let error: ResponsesMcpError = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(error, expected);
    assert_eq!(serde_json::to_value(&error).unwrap(), wire);
    let call: ResponsesOutputItem =
        serde_json::from_value(json!({"type":"mcp_call","error":wire})).unwrap();
    let ResponsesOutputItem::McpCall(call) = call else {
        panic!("expected call")
    };
    assert_eq!(call.error, Some(error));
}

#[rstest]
#[case::tools_not_array(json!({"type":"mcp_list_tools","tools":{}}))]
#[case::invalid_name(json!({"type":"mcp_list_tools","tools":[{"name":7}]}))]
#[case::invalid_schema(json!({"type":"mcp_list_tools","tools":[{"input_schema":{"properties":{"query":7}}}]}))]
#[case::invalid_error_code(json!({"type":"mcp_call","error":{"type":"http_error","code":"503","message":"unavailable"}}))]
#[case::missing_error_content(json!({"type":"mcp_call","error":{"type":"mcp_tool_execution_error"}}))]
#[case::unknown_error_tag(json!({"type":"mcp_call","error":{"type":"future"}}))]
fn mcp_output_rejects_malformed_known_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ResponsesOutputItem>(wire).is_err());
}
