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
    use rstest::rstest;

    use super::*;

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
}
