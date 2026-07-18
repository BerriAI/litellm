use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::responses::types::{ResponsesWsEvent, ResponsesWsEventType};
use serde_json::Value;

use crate::constants::DEFAULT_PROVIDER;
use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload, Usage,
};

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseSessionStatus {
    Success,
    Failure,
}

pub struct ResponsesWsStreaming {
    callbacks: Vec<Arc<dyn CustomLogger>>,
    litellm_call_id: String,
    id: String,
    model: String,
    usage: Usage,
    start_time: f64,
    end_time: f64,
    metadata: RequestMetadata,
}

impl ResponsesWsStreaming {
    pub fn new(
        callbacks: Vec<Arc<dyn CustomLogger>>,
        litellm_call_id: String,
        model: String,
        metadata: RequestMetadata,
    ) -> Self {
        let now = epoch_seconds();
        Self {
            callbacks,
            id: litellm_call_id.clone(),
            litellm_call_id,
            model,
            usage: Usage::default(),
            start_time: now,
            end_time: now,
            metadata,
        }
    }

    pub fn observe(&mut self, event: &ResponsesWsEvent) {
        match event.event_type {
            ResponsesWsEventType::ResponseCreated
            | ResponsesWsEventType::ResponseCompleted
            | ResponsesWsEventType::ResponseFailed
            | ResponsesWsEventType::ResponseIncomplete
            | ResponsesWsEventType::Error => self.observe_response(event),
            _ => {}
        }
    }

    fn observe_response(&mut self, event: &ResponsesWsEvent) {
        let response = event.data.get("response").and_then(Value::as_object);
        if let Some(id) = response
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            self.id = id.to_string();
            self.litellm_call_id = id.to_string();
        }
        if let Some(model) = response
            .and_then(|value| value.get("model"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            self.model = model.to_string();
        }
        let Some(usage) = response
            .and_then(|value| value.get("usage"))
            .and_then(Value::as_object)
        else {
            return;
        };
        if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
            self.usage.prompt_tokens += input;
        }
        if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
            self.usage.completion_tokens += output;
        }
        self.usage.total_tokens += usage
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

    fn build_payload(&self) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: self.id.clone(),
            litellm_call_id: self.litellm_call_id.clone(),
            call_type: "responses_websocket".to_string(),
            model: self.model.clone(),
            custom_llm_provider: DEFAULT_PROVIDER.to_string(),
            response_cost: 0.0,
            prompt_tokens: self.usage.prompt_tokens,
            completion_tokens: self.usage.completion_tokens,
            total_tokens: self.usage.total_tokens,
            start_time: self.start_time,
            end_time: self.end_time,
            stream: true,
            metadata: StandardLoggingMetadata {
                user_api_key_hash: self.metadata.user_api_key_hash.clone(),
                user_api_key_user_id: self.metadata.user_api_key_user_id.clone(),
                user_api_key_team_id: self.metadata.user_api_key_team_id.clone(),
                ..Default::default()
            },
            messages: None,
        }
    }

    pub async fn log_messages(&mut self, status: ResponseSessionStatus) {
        self.end_time = epoch_seconds();
        let payload = self.build_payload();
        let timing = CallbackTiming::new(payload.start_time, payload.end_time);
        let runner = CustomLoggerRunner::new(self.callbacks.clone());
        match status {
            ResponseSessionStatus::Success => {
                let response = CallbackValue::new("responses_websocket", Value::Null);
                let _ = runner
                    .async_log_success_event(
                        &ModelCallDetails::from_standard_logging_payload(payload),
                        &response,
                        timing,
                    )
                    .await;
            }
            ResponseSessionStatus::Failure => {
                let error = LoggingError {
                    message: "Responses WebSocket session ended in failure".to_string(),
                    kind: "ResponsesWebSocketError".to_string(),
                };
                let response = CallbackValue::new(
                    "error",
                    serde_json::json!({
                        "message": error.message,
                        "kind": error.kind,
                    }),
                );
                let _ = runner
                    .async_log_failure_event(
                        &ModelCallDetails::from_standard_logging_payload(payload)
                            .with_failure_error(error),
                        Some(&response),
                        timing,
                    )
                    .await;
            }
        }
    }
}
