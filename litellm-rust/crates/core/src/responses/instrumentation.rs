use crate::integrations::types::Usage;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType};
use serde_json::Value;
use std::sync::Mutex;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponsesWsObservation {
    pub(crate) model: String,
    pub(crate) usage: Usage,
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
}
