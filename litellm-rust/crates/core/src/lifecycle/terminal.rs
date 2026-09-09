use serde::Serialize;
use serde_json::Value;

use crate::integrations::custom_logger::{
    CallType, CallbackTiming, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload, Usage};

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CostInputs {
    pub response_cost: f64,
    pub metadata: StandardLoggingMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum TerminalClassification {
    Success,
    Cancelled { message: String },
    Incomplete { message: String },
    Failure { kind: String, message: String },
}

impl TerminalClassification {
    pub fn kind(&self) -> &str {
        match self {
            Self::Success => "Success",
            Self::Cancelled { .. } => "Cancelled",
            Self::Incomplete { .. } => "Incomplete",
            Self::Failure { kind, .. } => kind,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum RouteProjection {
    Ocr { value: Value },
    Messages { value: Value },
    ChatCompletions { value: Value },
    Audio { value: Value },
    Realtime { value: Value },
    ResponsesWs { value: Value },
}

impl RouteProjection {
    pub(crate) fn value(&self) -> &Value {
        match self {
            Self::Ocr { value }
            | Self::Messages { value }
            | Self::ChatCompletions { value }
            | Self::Audio { value }
            | Self::Realtime { value }
            | Self::ResponsesWs { value } => value,
        }
    }

    fn logging_input(&self) -> Option<Value> {
        match self {
            Self::Messages { value } | Self::ChatCompletions { value } => Some(value.clone()),
            Self::Ocr { .. }
            | Self::Audio { .. }
            | Self::Realtime { .. }
            | Self::ResponsesWs { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TerminalRecord {
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub call_type: String,
    pub model: String,
    pub provider: String,
    pub timing: CallbackTiming,
    pub usage: Usage,
    pub cost_inputs: CostInputs,
    pub classification: TerminalClassification,
    pub projection: RouteProjection,
}

impl From<&TerminalRecord> for StandardLoggingPayload {
    fn from(record: &TerminalRecord) -> Self {
        Self {
            id: record.call_id.clone(),
            litellm_call_id: record.call_id.clone(),
            call_type: record.call_type.clone(),
            model: record.model.clone(),
            custom_llm_provider: record.provider.clone(),
            response_cost: record.cost_inputs.response_cost,
            prompt_tokens: record.usage.prompt_tokens,
            completion_tokens: record.usage.completion_tokens,
            total_tokens: record.usage.total_tokens,
            start_time: record.timing.start_time,
            end_time: record.timing.end_time,
            stream: matches!(
                record.projection,
                RouteProjection::Realtime { .. } | RouteProjection::ResponsesWs { .. }
            ) || matches!(
                &record.projection,
                RouteProjection::Messages { value } | RouteProjection::ChatCompletions { value }
                    if value.get("stream").and_then(Value::as_bool) == Some(true)
            ),
            metadata: record.cost_inputs.metadata.clone(),
            messages: record.projection.logging_input(),
        }
    }
}

impl From<&TerminalRecord> for ModelCallDetails {
    fn from(record: &TerminalRecord) -> Self {
        let details = Self::from_standard_logging_payload(record.into());
        match &record.classification {
            TerminalClassification::Success => details,
            TerminalClassification::Cancelled { message } | TerminalClassification::Incomplete { message } => {
                details.with_failure_error(LoggingError {
                    kind: record.classification.kind().into(), message: message.clone(),
                })
            }
            TerminalClassification::Failure { kind, message } => {
                details.with_failure_error(LoggingError {
                    kind: kind.clone(),
                    message: message.clone(),
                })
            }
        }
    }
}

impl TerminalRecord {
    pub fn call_type(&self) -> CallType {
        CallType::from(self.call_type.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record() -> TerminalRecord {
        TerminalRecord {
            call_id: "call-1".to_string(),
            trace_id: Some("trace-1".to_string()),
            attempt: 2,
            call_type: "ocr".to_string(),
            model: "mistral-ocr-latest".to_string(),
            provider: "mistral".to_string(),
            timing: CallbackTiming::new(10.0, 11.5),
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            },
            cost_inputs: CostInputs {
                response_cost: 0.25,
                metadata: StandardLoggingMetadata {
                    user_api_key_user_id: Some("user-1".to_string()),
                    ..Default::default()
                },
            },
            classification: TerminalClassification::Success,
            projection: RouteProjection::Ocr {
                value: json!({"pages": [{"markdown": "ok"}]}),
            },
        }
    }

    #[test]
    fn terminal_record_projects_existing_logging_payload() {
        let record = record();
        let payload = StandardLoggingPayload::from(&record);
        let details = ModelCallDetails::from(&record);

        assert_eq!(payload.id, "call-1");
        assert_eq!(payload.litellm_call_id, "call-1");
        assert_eq!(payload.prompt_tokens, 3);
        assert_eq!(payload.response_cost, 0.25);
        assert_eq!(payload.messages, None);
        assert_eq!(
            details.standard_logging_payload.unwrap().litellm_call_id,
            "call-1"
        );
    }
}
