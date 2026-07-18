use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming};
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType};
use crate::{CoreError, CoreResult};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponsesWsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponsesWsMetadata {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesWsLogPayload {
    pub id: String,
    pub litellm_call_id: String,
    pub call_type: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub response_cost: f64,
    pub usage: ResponsesWsUsage,
    pub start_time: f64,
    pub end_time: f64,
    pub stream: bool,
    pub metadata: ResponsesWsMetadata,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponsesWsLogOutcome {
    Success {
        payload: ResponsesWsLogPayload,
        callback: ResponsesWsCallbackPayload,
    },
    Failure {
        payload: ResponsesWsLogPayload,
        callback: ResponsesWsCallbackPayload,
        error_message: String,
        error_kind: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesWsCallbackPayload {
    pub object: String,
    pub value: Value,
}

struct InstrumentationState {
    litellm_call_id: String,
    id: String,
    model: String,
    usage: ResponsesWsUsage,
    start_time: f64,
    end_time: f64,
    metadata: ResponsesWsMetadata,
    outcome: Option<ResponsesWsLogOutcome>,
}

pub struct ResponsesWsInstrumentation {
    state: Mutex<InstrumentationState>,
}

impl ResponsesWsInstrumentation {
    pub fn new(
        litellm_call_id: impl Into<String>,
        model: impl Into<String>,
        metadata: ResponsesWsMetadata,
    ) -> Self {
        let litellm_call_id = litellm_call_id.into();
        let now = epoch_seconds();
        Self {
            state: Mutex::new(InstrumentationState {
                id: litellm_call_id.clone(),
                litellm_call_id,
                model: model.into(),
                usage: ResponsesWsUsage::default(),
                start_time: now,
                end_time: now,
                metadata,
                outcome: None,
            }),
        }
    }

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
        if let Some(id) = response
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            state.id = id.to_string();
            state.litellm_call_id = id.to_string();
        }
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

    pub fn success_outcome(&self) -> ResponsesWsLogOutcome {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.end_time = epoch_seconds();
        ResponsesWsLogOutcome::Success {
            payload: build_payload(&state),
            callback: ResponsesWsCallbackPayload {
                object: "responses_websocket".to_string(),
                value: Value::Null,
            },
        }
    }

    pub fn failure_outcome(&self) -> ResponsesWsLogOutcome {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.end_time = epoch_seconds();
        ResponsesWsLogOutcome::Failure {
            payload: build_payload(&state),
            callback: ResponsesWsCallbackPayload {
                object: "error".to_string(),
                value: serde_json::json!({
                    "message": "Responses WebSocket session ended in failure",
                    "kind": "ResponsesWebSocketError",
                }),
            },
            error_message: "Responses WebSocket session ended in failure".to_string(),
            error_kind: "ResponsesWebSocketError".to_string(),
        }
    }

    pub fn take_outcome(&self) -> Option<ResponsesWsLogOutcome> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .outcome
            .take()
    }
}

type LifecycleFuture<'a, T> = Pin<Box<dyn Future<Output = CoreResult<T>> + Send + 'a>>;

impl CallLifecycleHooks<(), (), ()> for ResponsesWsInstrumentation {
    type PreCallFuture<'a> = LifecycleFuture<'a, ()>;
    type DuringCallFuture<'a> = LifecycleFuture<'a, ()>;
    type SuccessFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
    type FailureFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: (),
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move { Ok(request) })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: (),
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { Ok(request) })
    }

    fn async_log_success_event<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _response: &'a (),
        _timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a> {
        Box::pin(async move {
            let outcome = self.success_outcome();
            if let Ok(mut state) = self.state.lock() {
                state.outcome = Some(outcome);
            }
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a CoreError,
        _timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        Box::pin(async move {
            let outcome = self.failure_outcome();
            if let Ok(mut state) = self.state.lock() {
                state.outcome = Some(outcome);
            }
        })
    }
}

fn build_payload(state: &InstrumentationState) -> ResponsesWsLogPayload {
    ResponsesWsLogPayload {
        id: state.id.clone(),
        litellm_call_id: state.litellm_call_id.clone(),
        call_type: "responses_websocket".to_string(),
        model: state.model.clone(),
        custom_llm_provider: "openai".to_string(),
        response_cost: 0.0,
        usage: state.usage.clone(),
        start_time: state.start_time,
        end_time: state.end_time,
        stream: true,
        metadata: state.metadata.clone(),
    }
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid Responses WebSocket event")
    }

    #[test]
    fn accumulates_upstream_usage_and_identity() {
        let instrumentation =
            ResponsesWsInstrumentation::new("call-1", "gpt-5", ResponsesWsMetadata::default());
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

        let ResponsesWsLogOutcome::Success { payload, .. } = instrumentation.success_outcome()
        else {
            panic!("expected success outcome");
        };
        assert_eq!(payload.id, "resp-1");
        assert_eq!(payload.model, "gpt-5-mini");
        assert_eq!(payload.usage.prompt_tokens, 3);
        assert_eq!(payload.usage.completion_tokens, 5);
        assert_eq!(payload.usage.total_tokens, 8);
        assert!(payload.end_time >= payload.start_time);
    }

    #[test]
    fn builds_failure_payload_without_dispatching_callbacks() {
        let instrumentation =
            ResponsesWsInstrumentation::new("call-1", "gpt-5", ResponsesWsMetadata::default());
        assert!(matches!(
            instrumentation.failure_outcome(),
            ResponsesWsLogOutcome::Failure { .. }
        ));
    }
}
