use crate::integrations::types::Usage;
use crate::lifecycle::TerminalClassification;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType};
use serde_json::Value;
use std::sync::Mutex;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponsesWsObservation {
    pub(crate) model: String,
    pub(crate) usage: Usage,
    pub(crate) usage_available: bool,
}

#[derive(Default)]
pub struct ResponsesWsInstrumentation {
    state: Mutex<ResponsesWsObservation>,
}

impl ResponsesWsInstrumentation {
    pub fn observe(&self, event: &ResponsesWsEvent) {
        if !matches!(
            event.event_type,
            ResponsesWsEventType::ResponseCreated
                | ResponsesWsEventType::ResponseCompleted
                | ResponsesWsEventType::ResponseFailed
                | ResponsesWsEventType::ResponseIncomplete
                | ResponsesWsEventType::Error
        ) {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let Some(response) = event.data.get("response").and_then(Value::as_object) else {
            return;
        };
        if let Some(model) = response
            .get("model")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            state.model = model.to_string();
        }
        let Some(usage) = response.get("usage").and_then(Value::as_object) else {
            return;
        };
        state.usage_available = true;
        if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
            state.usage.prompt_tokens += input;
        }
        if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
            state.usage.completion_tokens += output;
        }
        state.usage.total_tokens += usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| {
                usage
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    + usage
                        .get("output_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
            });
    }

    pub fn snapshot(&self) -> ResponsesWsObservation {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn terminal_classification(
        &self,
        event: &ResponsesWsEvent,
    ) -> Option<TerminalClassification> {
        match event.event_type {
            ResponsesWsEventType::ResponseCompleted => Some(TerminalClassification::Success),
            ResponsesWsEventType::ResponseFailed => Some(provider_failure(
                "ResponseFailed",
                response_error_message(event).unwrap_or("provider response failed"),
            )),
            ResponsesWsEventType::ResponseIncomplete => Some(provider_failure(
                "ResponseIncomplete",
                incomplete_message(event).unwrap_or("provider response was incomplete"),
            )),
            ResponsesWsEventType::Error => Some(provider_failure(
                "ProviderError",
                top_level_error_message(event).unwrap_or("provider returned an error"),
            )),
            _ => None,
        }
    }
}

fn response_error_message(event: &ResponsesWsEvent) -> Option<&str> {
    event
        .data
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("error"))
        .and_then(Value::as_object)
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
}

fn incomplete_message(event: &ResponsesWsEvent) -> Option<&str> {
    event
        .data
        .get("response")
        .and_then(Value::as_object)
        .and_then(|response| response.get("incomplete_details"))
        .and_then(Value::as_object)
        .and_then(|details| details.get("reason"))
        .and_then(Value::as_str)
}

fn top_level_error_message(event: &ResponsesWsEvent) -> Option<&str> {
    event
        .data
        .get("error")
        .and_then(Value::as_object)
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
}

fn provider_failure(kind: &str, message: &str) -> TerminalClassification {
    let message = message.trim().chars().take(512).collect::<String>();
    TerminalClassification::Failure {
        kind: kind.to_string(),
        message: if message.is_empty() {
            "provider returned an unspecified failure".to_string()
        } else {
            message
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn event(value: Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid Responses WebSocket event")
    }

    #[test]
    fn accumulates_upstream_usage_and_identity() {
        let instrumentation = ResponsesWsInstrumentation::default();
        instrumentation.observe(&event(serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "gpt-5-mini",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8
                }
            }
        })));

        let observation = instrumentation.snapshot();
        assert_eq!(observation.model, "gpt-5-mini");
        assert_eq!(observation.usage.prompt_tokens, 3);
        assert_eq!(observation.usage.completion_tokens, 5);
        assert_eq!(observation.usage.total_tokens, 8);
    }

    #[test]
    fn classifies_provider_terminal_frames_without_serializing_the_frame() {
        let instrumentation = ResponsesWsInstrumentation::default();
        let cases = [
            (
                serde_json::json!({"type":"response.failed","response":{"error":{"message":"request rejected"}}}),
                "ResponseFailed",
                "request rejected",
            ),
            (
                serde_json::json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}),
                "ResponseIncomplete",
                "max_output_tokens",
            ),
            (
                serde_json::json!({"type":"error","error":{"type":"server_error","message":"provider unavailable"}}),
                "ProviderError",
                "provider unavailable",
            ),
        ];

        for (frame, expected_kind, expected_message) in cases {
            assert_eq!(
                instrumentation.terminal_classification(&event(frame)),
                Some(TerminalClassification::Failure {
                    kind: expected_kind.to_string(),
                    message: expected_message.to_string(),
                })
            );
        }
    }
}
