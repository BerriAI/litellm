use super::ResponsesOutputItem;
use crate::recognized::Recognized;
use serde_json::{Map, Value};

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
        formats::responses::streaming_websocket::ResponsesEventResponse, recognized::Recognized,
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
