use super::ResponsesOutputItem;
use crate::recognized::Recognized;
use serde_json::{Map, Value};

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::AsRefStr,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(with = "String"))]
pub enum ResponsesWsEventType {
    #[strum(serialize = "response.create")]
    ResponseCreate,
    #[strum(serialize = "response.created")]
    ResponseCreated,
    #[strum(serialize = "response.completed")]
    ResponseCompleted,
    #[strum(serialize = "response.failed")]
    ResponseFailed,
    #[strum(serialize = "response.incomplete")]
    ResponseIncomplete,
    #[strum(serialize = "error")]
    Error,
    #[strum(default, transparent)]
    Other(String),
}

impl ResponsesWsEventType {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ResponsesWsEvent {
    #[serde(rename = "type")]
    pub event_type: ResponsesWsEventType,
    #[serde(flatten)]
    pub data: Map<String, Value>,
}

impl ResponsesWsEvent {
    pub fn model(&self) -> Option<&str> {
        let model = self.data.get("model").and_then(Value::as_str);
        if model.is_some() {
            return model;
        }
        self.data
            .get("response")
            .and_then(Value::as_object)
            .and_then(|response| response.get("model"))
            .and_then(Value::as_str)
    }

    pub fn is_response_create(&self) -> bool {
        self.event_type == ResponsesWsEventType::ResponseCreate
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq)]
pub struct ResponsesErrorFrame {
    #[serde(rename = "type")]
    pub frame_type: &'static str,
    pub error: ResponsesErrorBody,
}

impl ResponsesErrorFrame {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            frame_type: "error",
            error: ResponsesErrorBody {
                error_type: "invalid_request_error",
                message: message.into(),
            },
        }
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq)]
pub struct ResponsesErrorBody {
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub message: String,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesEventResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub status: Option<String>,
    pub output: Option<Vec<Recognized<ResponsesOutputItem>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use crate::formats::responses::ResponsesOutputItem;

    use crate::{
        formats::responses::streaming_websocket::{ResponsesEventResponse, ResponsesWsEventType},
        recognized::Recognized,
        test_support::*,
    };

    use serde_json::{Value, json};

    #[test]
    fn error_frame_matches_proxy_shape() {
        let frame = ResponsesErrorFrame::invalid_request("missing model");
        assert_eq!(
            serde_json::to_value(frame).expect("serializable"),
            serde_json::json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "missing model"
                }
            })
        );
    }

    #[rstest]
    #[case::flat(serde_json::json!({"type":"response.create","model":"model-a"}), Some("model-a"))]
    #[case::nested(serde_json::json!({"type":"response.create","response":{"model":"model-b"}}), Some("model-b"))]
    fn model_reads_flat_and_nested_create_shapes(
        #[case] payload: serde_json::Value,
        #[case] expected: Option<&str>,
    ) {
        let event: ResponsesWsEvent = serde_json::from_value(payload).expect("valid event");
        assert_eq!(event.model(), expected);
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
    fn websocket_event_type_round_trips(
        #[case] wire: &str,
        #[case] expected: ResponsesWsEventType,
    ) {
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
        let [Recognized::Known(ResponsesOutputItem::FunctionCall(call))] = output.as_slice() else {
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
    #[case::wrong_model(json!({"model":7}))]
    #[case::wrong_status(json!({"status":false}))]
    #[case::wrong_output(json!({"output":{}}))]
    fn event_response_rejects_malformed_typed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<ResponsesEventResponse>(wire).is_err());
    }

    #[rstest]
    #[case::unknown_type(json!({"type":"future_item","id":"item_1","payload":[1,null]}))]
    #[case::missing_tag(json!({"id":"item_1"}))]
    #[case::malformed_known(json!({"type":"message","content":[{"type":"output_text","text":7}]}))]
    fn event_response_keeps_unmodeled_output_items_beside_typed_ones(#[case] item: Value) {
        let wire = json!({"output":[{"type":"function_call","name":"lookup"}, item.clone()]});
        let response: ResponsesEventResponse = serde_json::from_value(wire.clone()).unwrap();
        let Some(
            [
                Recognized::Known(ResponsesOutputItem::FunctionCall(call)),
                Recognized::Unrecognized(kept),
            ],
        ) = response.output.as_deref()
        else {
            panic!("expected one typed item and one preserved item");
        };
        assert_eq!(call.name.as_deref(), Some("lookup"));
        assert_eq!(kept, &item);
        assert_eq!(serde_json::to_value(response).unwrap(), wire);
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
    #[case::compaction(json!({"type":"compaction","id":"cmp_1","encrypted_content":"opaque"}))]
    #[case::approval(json!({"type":"mcp_approval_request","id":"apr_1","server_label":"s","name":"n","arguments":"{}"}))]
    #[case::shell(json!({"type":"shell_call","call_id":"call_1","action":{"commands":["ls"]}}))]
    fn event_response_types_documented_output_items(#[case] item: Value) {
        let wire = json!({"output":[item.clone()]});
        let response: ResponsesEventResponse = serde_json::from_value(wire.clone()).unwrap();
        let Some([Recognized::Known(known)]) = response.output.as_deref() else {
            panic!("expected one typed item");
        };
        assert_eq!(
            known,
            &serde_json::from_value::<ResponsesOutputItem>(item).unwrap()
        );
        assert_eq!(serde_json::to_value(response).unwrap(), wire);
    }
}
