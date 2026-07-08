//! `RealTimeStreaming` — the realtime logging collector.
//!
//! Mirrors Python `litellm.realtime_api.main.RealTimeStreaming`: it observes the
//! event stream in O(1) (never buffering frames), accumulating just the fields
//! the spend log needs (model, id, cumulative usage), then on session close
//! builds a `StandardLoggingPayload` and fans it out to every registered
//! `CustomLogger`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::realtime::types::RealtimeEvent;
use serde_json::Value;

use crate::constants::DEFAULT_PROVIDER;
use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload, Usage,
};

/// Current wall-clock time as epoch seconds (float), matching the Python
/// `startTime`/`endTime` contract.
fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Status of a finished realtime session, mapped to the callback record status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionStatus {
    Success,
    Failure,
}

/// Accumulates realtime session state and emits a logging payload on close.
pub struct RealTimeStreaming {
    callbacks: Vec<Arc<dyn CustomLogger>>,
    /// REQUEST-ID RULE: the SpendLogs `request_id` == the OpenAI realtime session
    /// id (`sess_…`), captured from `session.created`. Both `id` and
    /// `litellm_call_id` are set to that value so the Python writer logs the same
    /// id regardless of which field it reads. The gateway-generated `rt-…` id
    /// (the constructor seed) is only a fallback for sessions that fail before
    /// `session.created` arrives.
    litellm_call_id: String,
    /// See the request-id rule above — mirrors `litellm_call_id`.
    id: String,
    model: String,
    custom_llm_provider: String,
    usage: Usage,
    response_cost: f64,
    start_time: f64,
    end_time: f64,
    metadata: RequestMetadata,
    /// Count of logging callbacks that failed to enqueue (non-fatal).
    dropped: u64,
}

impl RealTimeStreaming {
    /// Create a collector for one session. `litellm_call_id` is the gateway's
    /// per-connection id; `model` is the requested model (a sane default until
    /// `session.created` reports the upstream model).
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
            custom_llm_provider: DEFAULT_PROVIDER.to_string(),
            usage: Usage::default(),
            response_cost: 0.0,
            start_time: now,
            end_time: now,
            metadata,
            dropped: 0,
        }
    }

    /// Number of logging callbacks that failed to enqueue so far (test/observ.).
    #[allow(dead_code)]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Observe one realtime event. O(1): updates accumulated state only; never
    /// buffers frames. Safe to call on every event in either direction.
    pub fn observe(&mut self, event: &RealtimeEvent) {
        match event.event_type.as_str() {
            "session.created" | "session.updated" => self.on_session(event),
            "response.done" => self.on_response_done(event),
            _ => {}
        }
    }

    /// `session.created` / `session.updated` → capture upstream id + model.
    /// Per the request-id rule, the OpenAI session id becomes BOTH `id` and
    /// `litellm_call_id`, replacing the gateway-generated fallback.
    fn on_session(&mut self, event: &RealtimeEvent) {
        let session = event.data.get("session").and_then(Value::as_object);
        if let Some(id) = session.and_then(|s| s.get("id")).and_then(Value::as_str) {
            if !id.is_empty() {
                self.id = id.to_string();
                self.litellm_call_id = id.to_string();
            }
        }
        if let Some(model) = session.and_then(|s| s.get("model")).and_then(Value::as_str) {
            if !model.is_empty() {
                self.model = model.to_string();
            }
        }
    }

    /// `response.done` → add this response's usage to the cumulative totals.
    fn on_response_done(&mut self, event: &RealtimeEvent) {
        let usage = event
            .data
            .get("response")
            .and_then(Value::as_object)
            .and_then(|r| r.get("usage"))
            .and_then(Value::as_object);
        let Some(usage) = usage else { return };

        let input = usage.get("input_tokens").and_then(Value::as_u64);
        let output = usage.get("output_tokens").and_then(Value::as_u64);
        let total = usage.get("total_tokens").and_then(Value::as_u64);

        if let Some(input) = input {
            self.usage.prompt_tokens += input;
        }
        if let Some(output) = output {
            self.usage.completion_tokens += output;
        }
        // Prefer the upstream-reported total; otherwise derive it.
        match total {
            Some(total) => self.usage.total_tokens += total,
            None => {
                self.usage.total_tokens += input.unwrap_or(0) + output.unwrap_or(0);
            }
        }
    }

    /// Set the per-session response cost ($). Cost computation is Python-side in
    /// the proxy; the gateway forwards 0.0 by default and lets the proxy price.
    /// Public API (exercised in tests) for the future path where the gateway
    /// prices realtime sessions itself.
    #[allow(dead_code)]
    pub fn set_response_cost(&mut self, cost: f64) {
        self.response_cost = cost;
    }

    /// Build the `StandardLoggingPayload` from accumulated state.
    pub fn build_payload(&self) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: self.id.clone(),
            litellm_call_id: self.litellm_call_id.clone(),
            call_type: "realtime".to_string(),
            model: self.model.clone(),
            custom_llm_provider: self.custom_llm_provider.clone(),
            response_cost: self.response_cost,
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

    /// Finish the session: stamp the end time and fan the payload out to every
    /// callback. On a logger enqueue error we bump a non-fatal counter (the
    /// realtime session has already ended; a dropped log must never propagate).
    pub async fn log_messages(&mut self, status: SessionStatus) {
        self.end_time = epoch_seconds();
        let payload = self.build_payload();
        let timing = CallbackTiming::new(payload.start_time, payload.end_time);
        let runner = CustomLoggerRunner::new(self.callbacks.clone());

        match status {
            SessionStatus::Success => {
                let response = CallbackValue::new("realtime", serde_json::Value::Null);
                let report = runner
                    .async_log_success_event(
                        &ModelCallDetails::from_standard_logging_payload(payload),
                        &response,
                        timing,
                    )
                    .await;
                self.dropped += report.dropped as u64;
            }
            SessionStatus::Failure => {
                let error = LoggingError {
                    message: "realtime session ended in failure".to_string(),
                    kind: "RealtimeSessionError".to_string(),
                };
                let response = CallbackValue::new(
                    "error",
                    serde_json::json!({
                        "message": error.message,
                        "kind": error.kind,
                    }),
                );
                let report = runner
                    .async_log_failure_event(
                        &ModelCallDetails::from_standard_logging_payload(payload)
                            .with_failure_error(error),
                        Some(&response),
                        timing,
                    )
                    .await;
                self.dropped += report.dropped as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::LogError;
    use crate::integrations::custom_logger::LogFuture;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn event(raw: &str) -> RealtimeEvent {
        serde_json::from_str(raw).expect("valid event json")
    }

    /// A test logger that records the last payload it saw.
    #[derive(Default)]
    struct CapturingLogger {
        calls: AtomicU64,
        last_model: std::sync::Mutex<Option<String>>,
        last_total_tokens: AtomicU64,
    }

    impl CustomLogger for CapturingLogger {
        fn async_log_success_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            _response_obj: &'a CallbackValue,
            _timing: CallbackTiming,
        ) -> LogFuture<'a> {
            Box::pin(async move {
                let payload = model_call_details
                    .standard_logging_payload
                    .as_ref()
                    .expect("standard logging payload");
                self.calls.fetch_add(1, Ordering::SeqCst);
                *self.last_model.lock().unwrap() = Some(payload.model.clone());
                self.last_total_tokens
                    .store(payload.total_tokens, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn observe_accumulates_model_and_tokens_then_logs() {
        let logger = Arc::new(CapturingLogger::default());
        let callbacks: Vec<Arc<dyn CustomLogger>> = vec![logger.clone()];
        let mut streaming = RealTimeStreaming::new(
            callbacks,
            "call_abc".to_string(),
            "gpt-realtime".to_string(),
            RequestMetadata {
                user_api_key_hash: Some("hash123".to_string()),
                user_api_key_user_id: Some("user-1".to_string()),
                user_api_key_team_id: Some("team-1".to_string()),
            },
        );

        streaming.observe(&event(
            r#"{"type":"session.created","session":{"id":"sess_001","model":"gpt-realtime-2025"}}"#,
        ));
        streaming.observe(&event(
            r#"{"type":"response.done","response":{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#,
        ));
        // A second response.done accumulates.
        streaming.observe(&event(
            r#"{"type":"response.done","response":{"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}"#,
        ));

        let payload = streaming.build_payload();
        assert_eq!(payload.model, "gpt-realtime-2025");
        // Request-id rule: session.created's id becomes BOTH id and
        // litellm_call_id (replacing the "call_abc" gateway fallback), so the
        // SpendLogs request_id is always the OpenAI session id.
        assert_eq!(payload.id, "sess_001");
        assert_eq!(payload.litellm_call_id, "sess_001");
        assert_eq!(payload.prompt_tokens, 13);
        assert_eq!(payload.completion_tokens, 7);
        assert_eq!(payload.total_tokens, 20);
        assert_eq!(payload.response_cost, 0.0);
        assert_eq!(payload.call_type, "realtime");
        assert_eq!(payload.custom_llm_provider, "openai");
        assert_eq!(
            payload.metadata.user_api_key_hash.as_deref(),
            Some("hash123")
        );

        streaming.log_messages(SessionStatus::Success).await;
        assert_eq!(logger.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            logger.last_model.lock().unwrap().as_deref(),
            Some("gpt-realtime-2025")
        );
        assert_eq!(logger.last_total_tokens.load(Ordering::SeqCst), 20);
        assert_eq!(streaming.dropped(), 0);
    }

    #[test]
    fn payload_serializes_with_camelcase_times_and_realtime_call_type() {
        let mut streaming = RealTimeStreaming::new(
            Vec::new(),
            "call_xyz".to_string(),
            "gpt-realtime".to_string(),
            RequestMetadata::default(),
        );
        streaming.observe(&event(
            r#"{"type":"response.done","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
        ));
        streaming.set_response_cost(0.0042);
        let payload = streaming.build_payload();
        let json = serde_json::to_string(&payload).expect("serialize payload");

        assert!(json.contains("\"startTime\""), "missing startTime: {json}");
        assert!(json.contains("\"endTime\""), "missing endTime: {json}");
        assert!(
            json.contains("\"call_type\":\"realtime\""),
            "missing call_type realtime: {json}"
        );
        assert!(
            json.contains("\"response_cost\""),
            "missing response_cost: {json}"
        );
        assert_eq!(payload.response_cost, 0.0042);
    }

    /// A logger whose enqueue always fails should bump the dropped counter, not
    /// panic or propagate.
    #[tokio::test]
    async fn failing_logger_bumps_dropped_counter() {
        struct FailingLogger;
        impl CustomLogger for FailingLogger {
            fn async_log_success_event<'a>(
                &'a self,
                _model_call_details: &'a ModelCallDetails,
                _response_obj: &'a CallbackValue,
                _timing: CallbackTiming,
            ) -> LogFuture<'a> {
                Box::pin(async { Err(LogError::channel_full()) })
            }

            fn async_log_failure_event<'a>(
                &'a self,
                _model_call_details: &'a ModelCallDetails,
                _response_obj: Option<&'a CallbackValue>,
                _timing: CallbackTiming,
            ) -> LogFuture<'a> {
                Box::pin(async { Err(LogError::channel_closed()) })
            }
        }
        let callbacks: Vec<Arc<dyn CustomLogger>> = vec![Arc::new(FailingLogger)];
        let mut streaming = RealTimeStreaming::new(
            callbacks,
            "call_1".to_string(),
            "gpt-realtime".to_string(),
            RequestMetadata::default(),
        );
        streaming.log_messages(SessionStatus::Success).await;
        assert_eq!(streaming.dropped(), 1);
    }
}
