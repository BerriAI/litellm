use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::streaming::{JsonObject, ProviderCallContext};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponsesWsEventType {
    ResponseCreate,
    ResponseCreated,
    ResponseInProgress,
    ResponseReasoningSummaryPartAdded,
    ResponseReasoningSummaryTextDelta,
    ResponseReasoningSummaryTextDone,
    ResponseReasoningSummaryPartDone,
    ResponseOutputItemAdded,
    ResponseOutputTextDelta,
    ResponseOutputTextAnnotationAdded,
    ResponseOutputTextDone,
    ResponseRefusalDelta,
    ResponseRefusalDone,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    ResponseFileSearchCallInProgress,
    ResponseFileSearchCallSearching,
    ResponseFileSearchCallCompleted,
    ResponseWebSearchCallInProgress,
    ResponseWebSearchCallSearching,
    ResponseWebSearchCallCompleted,
    ResponseMcpListToolsInProgress,
    ResponseMcpListToolsCompleted,
    ResponseMcpListToolsFailed,
    ResponseMcpCallInProgress,
    ResponseMcpCallArgumentsDelta,
    ResponseMcpCallArgumentsDone,
    ResponseMcpCallCompleted,
    ResponseMcpCallFailed,
    ResponseContentPartAdded,
    ResponseContentPartDone,
    ResponseOutputItemDone,
    ResponseCompleted,
    ResponseFailed,
    ResponseIncomplete,
    ImageGenerationPartialImage,
    Error,
    Other(String),
}

impl ResponsesWsEventType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ResponseCreate => "response.create",
            Self::ResponseCreated => "response.created",
            Self::ResponseInProgress => "response.in_progress",
            Self::ResponseReasoningSummaryPartAdded => "response.reasoning_summary_part.added",
            Self::ResponseReasoningSummaryTextDelta => "response.reasoning_summary_text.delta",
            Self::ResponseReasoningSummaryTextDone => "response.reasoning_summary_text.done",
            Self::ResponseReasoningSummaryPartDone => "response.reasoning_summary_part.done",
            Self::ResponseOutputItemAdded => "response.output_item.added",
            Self::ResponseOutputTextDelta => "response.output_text.delta",
            Self::ResponseOutputTextAnnotationAdded => "response.output_text.annotation.added",
            Self::ResponseOutputTextDone => "response.output_text.done",
            Self::ResponseRefusalDelta => "response.refusal.delta",
            Self::ResponseRefusalDone => "response.refusal.done",
            Self::ResponseFunctionCallArgumentsDelta => "response.function_call_arguments.delta",
            Self::ResponseFunctionCallArgumentsDone => "response.function_call_arguments.done",
            Self::ResponseFileSearchCallInProgress => "response.file_search_call.in_progress",
            Self::ResponseFileSearchCallSearching => "response.file_search_call.searching",
            Self::ResponseFileSearchCallCompleted => "response.file_search_call.completed",
            Self::ResponseWebSearchCallInProgress => "response.web_search_call.in_progress",
            Self::ResponseWebSearchCallSearching => "response.web_search_call.searching",
            Self::ResponseWebSearchCallCompleted => "response.web_search_call.completed",
            Self::ResponseMcpListToolsInProgress => "response.mcp_list_tools.in_progress",
            Self::ResponseMcpListToolsCompleted => "response.mcp_list_tools.completed",
            Self::ResponseMcpListToolsFailed => "response.mcp_list_tools.failed",
            Self::ResponseMcpCallInProgress => "response.mcp_call.in_progress",
            Self::ResponseMcpCallArgumentsDelta => "response.mcp_call_arguments.delta",
            Self::ResponseMcpCallArgumentsDone => "response.mcp_call_arguments.done",
            Self::ResponseMcpCallCompleted => "response.mcp_call.completed",
            Self::ResponseMcpCallFailed => "response.mcp_call.failed",
            Self::ResponseContentPartAdded => "response.content_part.added",
            Self::ResponseContentPartDone => "response.content_part.done",
            Self::ResponseOutputItemDone => "response.output_item.done",
            Self::ResponseCompleted => "response.completed",
            Self::ResponseFailed => "response.failed",
            Self::ResponseIncomplete => "response.incomplete",
            Self::ImageGenerationPartialImage => "image_generation.partial_image",
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
            "response.in_progress" => Self::ResponseInProgress,
            "response.reasoning_summary_part.added" => Self::ResponseReasoningSummaryPartAdded,
            "response.reasoning_summary_text.delta" => Self::ResponseReasoningSummaryTextDelta,
            "response.reasoning_summary_text.done" => Self::ResponseReasoningSummaryTextDone,
            "response.reasoning_summary_part.done" => Self::ResponseReasoningSummaryPartDone,
            "response.output_item.added" => Self::ResponseOutputItemAdded,
            "response.output_text.delta" => Self::ResponseOutputTextDelta,
            "response.output_text.annotation.added" => Self::ResponseOutputTextAnnotationAdded,
            "response.output_text.done" => Self::ResponseOutputTextDone,
            "response.refusal.delta" => Self::ResponseRefusalDelta,
            "response.refusal.done" => Self::ResponseRefusalDone,
            "response.function_call_arguments.delta" => Self::ResponseFunctionCallArgumentsDelta,
            "response.function_call_arguments.done" => Self::ResponseFunctionCallArgumentsDone,
            "response.file_search_call.in_progress" => Self::ResponseFileSearchCallInProgress,
            "response.file_search_call.searching" => Self::ResponseFileSearchCallSearching,
            "response.file_search_call.completed" => Self::ResponseFileSearchCallCompleted,
            "response.web_search_call.in_progress" => Self::ResponseWebSearchCallInProgress,
            "response.web_search_call.searching" => Self::ResponseWebSearchCallSearching,
            "response.web_search_call.completed" => Self::ResponseWebSearchCallCompleted,
            "response.mcp_list_tools.in_progress" => Self::ResponseMcpListToolsInProgress,
            "response.mcp_list_tools.completed" => Self::ResponseMcpListToolsCompleted,
            "response.mcp_list_tools.failed" => Self::ResponseMcpListToolsFailed,
            "response.mcp_call.in_progress" => Self::ResponseMcpCallInProgress,
            "response.mcp_call_arguments.delta" => Self::ResponseMcpCallArgumentsDelta,
            "response.mcp_call_arguments.done" => Self::ResponseMcpCallArgumentsDone,
            "response.mcp_call.completed" => Self::ResponseMcpCallCompleted,
            "response.mcp_call.failed" => Self::ResponseMcpCallFailed,
            "response.content_part.added" => Self::ResponseContentPartAdded,
            "response.content_part.done" => Self::ResponseContentPartDone,
            "response.output_item.done" => Self::ResponseOutputItemDone,
            "response.completed" => Self::ResponseCompleted,
            "response.failed" => Self::ResponseFailed,
            "response.incomplete" => Self::ResponseIncomplete,
            "image_generation.partial_image" => Self::ImageGenerationPartialImage,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesWsTransformResult {
    pub events: Vec<ResponsesWsEvent>,
}

pub type ResponsesStreamEvent = ResponsesWsEvent;
pub type ResponseCommand = ResponsesWsEvent;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<JsonObject>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Name(String),
    Definition(JsonObject),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesStreamRequestBody {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<JsonObject>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

pub struct ResponsesStreamRequest {
    pub body: ResponsesStreamRequestBody,
    pub context: ProviderCallContext,
}

pub struct ResponsesWebSocketRequest {
    pub context: ProviderCallContext,
}

impl ResponsesWsTransformResult {
    pub fn passthrough(event: ResponsesWsEvent) -> Self {
        Self {
            events: vec![event],
        }
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
    use super::*;

    #[test]
    fn event_type_round_trips_known_and_unknown_values() {
        let known: ResponsesWsEventType =
            serde_json::from_str("\"response.completed\"").expect("valid event type");
        assert_eq!(known, ResponsesWsEventType::ResponseCompleted);
        let output_delta: ResponsesWsEventType =
            serde_json::from_str("\"response.output_text.delta\"").expect("valid event type");
        assert_eq!(output_delta, ResponsesWsEventType::ResponseOutputTextDelta);
        let unknown: ResponsesWsEventType =
            serde_json::from_str("\"response.future_event\"").expect("valid event type");
        assert_eq!(
            unknown,
            ResponsesWsEventType::Other("response.future_event".to_string())
        );
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

    #[test]
    fn model_reads_flat_and_nested_create_shapes() {
        let flat: ResponsesWsEvent =
            serde_json::from_value(serde_json::json!({"type":"response.create","model":"gpt-5"}))
                .expect("valid event");
        let nested: ResponsesWsEvent = serde_json::from_value(serde_json::json!({
            "type":"response.create",
            "response":{"model":"gpt-5-mini"}
        }))
        .expect("valid event");
        assert_eq!(flat.model(), Some("gpt-5"));
        assert_eq!(nested.model(), Some("gpt-5-mini"));
    }

    #[test]
    fn stream_request_deserializes_the_public_responses_shape() {
        let request: ResponsesStreamRequestBody = serde_json::from_value(serde_json::json!({
            "model": "gpt-5",
            "input": "hello",
            "stream": true,
            "max_output_tokens": 32
        }))
        .expect("public request shape");

        assert_eq!(request.model, "gpt-5");
        assert_eq!(request.stream, Some(true));
        assert!(matches!(request.input, ResponsesInput::Text(ref text) if text == "hello"));
    }

    #[test]
    fn unknown_stream_events_round_trip_for_forward_compatibility() {
        let event: ResponsesStreamEvent = serde_json::from_value(serde_json::json!({
            "type": "response.future_event",
            "sequence_number": 7,
            "future_field": "value"
        }))
        .expect("unknown event");

        assert_eq!(
            event.event_type,
            ResponsesWsEventType::Other("response.future_event".to_string())
        );
        assert_eq!(
            serde_json::to_value(event).expect("serializable event"),
            serde_json::json!({
                "type": "response.future_event",
                "sequence_number": 7,
                "future_field": "value"
            })
        );
    }
}
