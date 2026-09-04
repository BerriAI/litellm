use std::collections::BTreeMap;
use std::convert::Infallible;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CallbackDecision {
    Unchanged,
    Replace {
        payload: Value,
    },
    Reject {
        message: String,
        status_code: Option<u16>,
    },
}

#[derive(Clone, Serialize)]
pub struct ProviderPreCall {
    pub provider: String,
    pub model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub started_at: f64,
    pub request: BTreeMap<String, Value>,
    pub api_base: String,
    pub headers: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct ProviderPostCall {
    pub provider: String,
    pub model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub started_at: f64,
    pub response: Value,
    pub status_code: u16,
    pub headers: BTreeMap<String, String>,
    pub ended_at: f64,
}

#[derive(Serialize)]
pub struct ProviderError {
    pub provider: String,
    pub model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub started_at: f64,
    pub message: String,
    pub stage: &'static str,
    pub committed: bool,
    pub status_code: Option<u16>,
    pub will_retry: bool,
    pub ended_at: f64,
}

#[derive(Serialize)]
pub struct ProviderStreamEvent {
    pub provider: String,
    pub model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub started_at: f64,
    pub event: Value,
    pub sequence: u64,
}

#[derive(Serialize)]
pub struct ProviderStreamClose {
    pub provider: String,
    pub model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub started_at: f64,
    pub outcome: String,
    pub ended_at: f64,
}

#[derive(Serialize)]
pub struct SessionEvent {
    pub session_id: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub event: Option<Value>,
    pub response_id: Option<String>,
    pub sequence: Option<u64>,
    pub message: Option<String>,
}

#[macro_export]
macro_rules! provider_attempt_observer_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                pre_call: PreCall($crate::provider_callbacks::ProviderPreCall) -> $crate::provider_callbacks::CallbackDecision = direct;
                post_call: PostCall($crate::provider_callbacks::ProviderPostCall) -> $crate::provider_callbacks::CallbackDecision = direct;
                error: Error($crate::provider_callbacks::ProviderError) -> () = direct;
            }
        }
    };
}

#[macro_export]
macro_rules! streaming_observer_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                pre_call: StreamingPreCall($crate::provider_callbacks::ProviderPreCall) -> $crate::provider_callbacks::CallbackDecision = direct;
                post_call: StreamingPostCall($crate::provider_callbacks::ProviderPostCall) -> $crate::provider_callbacks::CallbackDecision = direct;
                error: StreamingError($crate::provider_callbacks::ProviderError) -> () = direct;
                stream_event: StreamEvent($crate::provider_callbacks::ProviderStreamEvent) -> $crate::provider_callbacks::CallbackDecision = direct;
                stream_close: StreamClose($crate::provider_callbacks::ProviderStreamClose) -> () = direct;
            }
        }
    };
}

#[macro_export]
macro_rules! session_observer_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                before_connect: BeforeConnect($crate::provider_callbacks::SessionEvent) -> $crate::provider_callbacks::CallbackDecision = awaitable;
                connected: Connected($crate::provider_callbacks::SessionEvent) -> () = awaitable;
                before_send: BeforeSend($crate::provider_callbacks::SessionEvent) -> $crate::provider_callbacks::CallbackDecision = awaitable;
                after_receive: AfterReceive($crate::provider_callbacks::SessionEvent) -> $crate::provider_callbacks::CallbackDecision = awaitable;
                response_complete: ResponseComplete($crate::provider_callbacks::SessionEvent) -> () = awaitable;
                response_error: ResponseError($crate::provider_callbacks::SessionEvent) -> () = awaitable;
                error: SessionError($crate::provider_callbacks::SessionEvent) -> () = awaitable;
                close: Close($crate::provider_callbacks::SessionEvent) -> () = awaitable;
            }
        }
    };
}

provider_attempt_observer_catalog!(crate::define_hooks, pub trait ProviderAttemptObserver;);
streaming_observer_catalog!(crate::define_hooks, pub trait StreamingObserver;);
session_observer_catalog!(crate::define_hooks, pub trait SessionObserver;);

pub struct NoopProviderAttemptObserver;

impl ProviderAttemptObserver for NoopProviderAttemptObserver {
    type Error = Infallible;

    async fn pre_call(&mut self, _input: &ProviderPreCall) -> Result<CallbackDecision, Infallible> {
        Ok(CallbackDecision::Unchanged)
    }

    async fn post_call(
        &mut self,
        _input: &ProviderPostCall,
    ) -> Result<CallbackDecision, Infallible> {
        Ok(CallbackDecision::Unchanged)
    }

    async fn error(&mut self, _input: &ProviderError) -> Result<(), Infallible> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::CallbackDecision;

    #[test]
    fn callback_decisions_have_a_tagged_wire_contract() {
        assert_eq!(
            serde_json::from_value::<CallbackDecision>(json!({"action": "unchanged"})).unwrap(),
            CallbackDecision::Unchanged
        );
        assert_eq!(
            serde_json::from_value::<CallbackDecision>(
                json!({"action": "replace", "payload": {"masked": true}})
            )
            .unwrap(),
            CallbackDecision::Replace {
                payload: json!({"masked": true})
            }
        );
        assert_eq!(
            serde_json::from_value::<CallbackDecision>(json!({
                "action": "reject",
                "message": "blocked",
                "status_code": 400
            }))
            .unwrap(),
            CallbackDecision::Reject {
                message: "blocked".to_string(),
                status_code: Some(400)
            }
        );
    }
}
