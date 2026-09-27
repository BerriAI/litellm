use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponsesWsEventType {
    ResponseCreate,
    ResponseCreated,
    ResponseCompleted,
    ResponseFailed,
    ResponseIncomplete,
    Error,
    Other(String),
}

impl ResponsesWsEventType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ResponseCreate => "response.create",
            Self::ResponseCreated => "response.created",
            Self::ResponseCompleted => "response.completed",
            Self::ResponseFailed => "response.failed",
            Self::ResponseIncomplete => "response.incomplete",
            Self::Error => "error",
            Self::Other(value) => value,
        }
    }
}

impl Serialize for ResponsesWsEventType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ResponsesWsEventType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "response.create" => Self::ResponseCreate,
            "response.created" => Self::ResponseCreated,
            "response.completed" => Self::ResponseCompleted,
            "response.failed" => Self::ResponseFailed,
            "response.incomplete" => Self::ResponseIncomplete,
            "error" => Self::Error,
            _ => Self::Other(value),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponsesErrorBody {
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::known("response.completed", ResponsesWsEventType::ResponseCompleted)]
    #[case::unknown(
        "response.output_text.delta",
        ResponsesWsEventType::Other("response.output_text.delta".to_string())
    )]
    fn event_type_round_trips_known_and_unknown_values(
        #[case] value: &str,
        #[case] expected: ResponsesWsEventType,
    ) {
        let actual: ResponsesWsEventType =
            serde_json::from_str(&serde_json::to_string(value).unwrap()).expect("valid event type");
        assert_eq!(actual, expected);
    }

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
