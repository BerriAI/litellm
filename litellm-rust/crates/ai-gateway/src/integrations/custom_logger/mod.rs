//! The `CustomLogger` trait — the Rust mirror of Python
//! `litellm/integrations/custom_logger.py::CustomLogger`.
//!
//! The Python-named async terminal methods are the public Rust callback shape.

use std::sync::Arc;

pub mod types;

pub use types::{
    CallType, CallbackDispatchReport, CallbackTiming, CallbackValue, LogError, LogFuture,
    LoggingError, ModelCallDetails,
};

pub trait CustomLogger: Send + Sync {
    /// Python 1:1 name: `async_log_success_event(model_call_details, response_obj, start_time, end_time)`.
    fn async_log_success_event<'a>(
        &'a self,
        _model_call_details: &'a ModelCallDetails,
        _response_obj: &'a CallbackValue,
        _timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async { Ok(()) })
    }

    /// Python 1:1 name: `async_log_failure_event(model_call_details, response_obj, start_time, end_time)`.
    fn async_log_failure_event<'a>(
        &'a self,
        _model_call_details: &'a ModelCallDetails,
        _response_obj: Option<&'a CallbackValue>,
        _timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

pub struct CustomLoggerRunner {
    loggers: Vec<Arc<dyn CustomLogger>>,
}

impl CustomLoggerRunner {
    pub fn new(loggers: Vec<Arc<dyn CustomLogger>>) -> Self {
        Self { loggers }
    }

    pub fn is_empty(&self) -> bool {
        self.loggers.is_empty()
    }

    pub async fn async_log_success_event(
        &self,
        model_call_details: &ModelCallDetails,
        response_obj: &CallbackValue,
        timing: CallbackTiming,
    ) -> CallbackDispatchReport {
        if self.loggers.is_empty() {
            return CallbackDispatchReport::default();
        }

        let mut report = CallbackDispatchReport::default();
        for logger in &self.loggers {
            report.invoked += 1;
            if let Err(err) = logger
                .async_log_success_event(model_call_details, response_obj, timing)
                .await
            {
                report.dropped += 1;
                eprintln!("litellm-ai-gateway: async_log_success_event dropped: {err}");
            }
        }
        report
    }

    pub async fn async_log_failure_event(
        &self,
        model_call_details: &ModelCallDetails,
        response_obj: Option<&CallbackValue>,
        timing: CallbackTiming,
    ) -> CallbackDispatchReport {
        if self.loggers.is_empty() {
            return CallbackDispatchReport::default();
        }

        let mut report = CallbackDispatchReport::default();
        for logger in &self.loggers {
            report.invoked += 1;
            if let Err(err) = logger
                .async_log_failure_event(model_call_details, response_obj, timing)
                .await
            {
                report.dropped += 1;
                eprintln!("litellm-ai-gateway: async_log_failure_event dropped: {err}");
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;
    use std::sync::Mutex;

    #[derive(Clone, Debug, PartialEq)]
    struct RecordedEvent {
        hook: &'static str,
        model: String,
        provider: String,
        call_type: String,
        request_id: Option<String>,
        litellm_call_id: Option<String>,
        user_id: Option<String>,
        response_object: Option<String>,
        error_kind: Option<String>,
        start_time: f64,
        end_time: f64,
        standard_logging_model: Option<String>,
    }

    #[derive(Default)]
    struct RecordingCustomLogger {
        events: Mutex<Vec<RecordedEvent>>,
    }

    impl RecordingCustomLogger {
        fn events(&self) -> Vec<RecordedEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl CustomLogger for RecordingCustomLogger {
        fn async_log_success_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            response_obj: &'a CallbackValue,
            timing: CallbackTiming,
        ) -> LogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push(RecordedEvent {
                    hook: "async_log_success_event",
                    model: model_call_details.model.clone(),
                    provider: model_call_details.custom_llm_provider.clone(),
                    call_type: model_call_details.call_type.to_string(),
                    request_id: model_call_details.request_id.clone(),
                    litellm_call_id: model_call_details.litellm_call_id.clone(),
                    user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                    response_object: Some(response_obj.object.clone()),
                    error_kind: None,
                    start_time: timing.start_time,
                    end_time: timing.end_time,
                    standard_logging_model: model_call_details
                        .standard_logging_payload
                        .as_ref()
                        .map(|payload| payload.model.clone()),
                });
                Ok(())
            })
        }

        fn async_log_failure_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            response_obj: Option<&'a CallbackValue>,
            timing: CallbackTiming,
        ) -> LogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push(RecordedEvent {
                    hook: "async_log_failure_event",
                    model: model_call_details.model.clone(),
                    provider: model_call_details.custom_llm_provider.clone(),
                    call_type: model_call_details.call_type.to_string(),
                    request_id: model_call_details.request_id.clone(),
                    litellm_call_id: model_call_details.litellm_call_id.clone(),
                    user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                    response_object: response_obj.map(|value| value.object.clone()),
                    error_kind: model_call_details
                        .failure_error
                        .as_ref()
                        .map(|error| error.kind.clone()),
                    start_time: timing.start_time,
                    end_time: timing.end_time,
                    standard_logging_model: model_call_details
                        .standard_logging_payload
                        .as_ref()
                        .map(|payload| payload.model.clone()),
                });
                Ok(())
            })
        }
    }

    fn payload(call_type: &str, model: &str, provider: &str) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: format!("req_{call_type}"),
            litellm_call_id: format!("call_{call_type}"),
            call_type: call_type.to_string(),
            model: model.to_string(),
            custom_llm_provider: provider.to_string(),
            response_cost: 0.25,
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 7,
            start_time: 10.0,
            end_time: 11.5,
            stream: false,
            metadata: StandardLoggingMetadata {
                user_api_key_hash: Some("hash".to_string()),
                user_api_key_user_id: Some("user".to_string()),
                user_api_key_team_id: Some("team".to_string()),
                ..Default::default()
            },
            messages: Some(json!([{"role": "user", "content": "read this"}])),
        }
    }

    #[tokio::test]
    async fn rust_custom_logger_reads_success_payload_for_ocr() {
        let logger = Arc::new(RecordingCustomLogger::default());
        let runner = CustomLoggerRunner::new(vec![logger.clone()]);
        let details = ModelCallDetails::from_standard_logging_payload(payload(
            "ocr",
            "mistral-ocr-latest",
            "mistral",
        ));
        let response = CallbackValue::new("ocr", json!({"pages": [{"markdown": "ok"}]}));
        let report = runner
            .async_log_success_event(&details, &response, CallbackTiming::new(10.0, 11.5))
            .await;

        assert_eq!(report.invoked, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(
            logger.events(),
            vec![RecordedEvent {
                hook: "async_log_success_event",
                model: "mistral-ocr-latest".to_string(),
                provider: "mistral".to_string(),
                call_type: "ocr".to_string(),
                request_id: Some("req_ocr".to_string()),
                litellm_call_id: Some("call_ocr".to_string()),
                user_id: Some("user".to_string()),
                response_object: Some("ocr".to_string()),
                error_kind: None,
                start_time: 10.0,
                end_time: 11.5,
                standard_logging_model: Some("mistral-ocr-latest".to_string()),
            }]
        );
    }

    #[tokio::test]
    async fn rust_custom_logger_reads_failure_payload_for_non_ocr_call_type() {
        let logger = Arc::new(RecordingCustomLogger::default());
        let runner = CustomLoggerRunner::new(vec![logger.clone()]);
        let details = ModelCallDetails::from_standard_logging_payload(payload(
            "acompletion",
            "gpt-4.1-mini",
            "openai",
        ))
        .with_failure_error(LoggingError {
            message: "provider failed".to_string(),
            kind: "ProviderError".to_string(),
        });
        let response = CallbackValue::new("error", json!({"message": "provider failed"}));
        let report = runner
            .async_log_failure_event(&details, Some(&response), CallbackTiming::new(2.0, 3.0))
            .await;

        assert_eq!(report.invoked, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(
            logger.events(),
            vec![RecordedEvent {
                hook: "async_log_failure_event",
                model: "gpt-4.1-mini".to_string(),
                provider: "openai".to_string(),
                call_type: "acompletion".to_string(),
                request_id: Some("req_acompletion".to_string()),
                litellm_call_id: Some("call_acompletion".to_string()),
                user_id: Some("user".to_string()),
                response_object: Some("error".to_string()),
                error_kind: Some("ProviderError".to_string()),
                start_time: 2.0,
                end_time: 3.0,
                standard_logging_model: Some("gpt-4.1-mini".to_string()),
            }]
        );
    }

    #[tokio::test]
    async fn no_callback_fast_path_dispatches_nothing() {
        let runner = CustomLoggerRunner::new(Vec::new());
        let details = ModelCallDetails::new("mistral-ocr-latest", "mistral", CallType::Ocr);
        let response = CallbackValue::new("ocr", json!({}));

        let report = runner
            .async_log_success_event(&details, &response, CallbackTiming::new(1.0, 1.5))
            .await;

        assert!(runner.is_empty());
        assert_eq!(report, CallbackDispatchReport::default());
    }

    #[test]
    fn with_standard_logging_payload_keeps_top_level_fields_in_sync() {
        let details = ModelCallDetails::new("old-model", "old-provider", CallType::Completion)
            .with_standard_logging_payload(payload("ocr", "mistral-ocr-latest", "mistral"));

        assert_eq!(details.model, "mistral-ocr-latest");
        assert_eq!(details.custom_llm_provider, "mistral");
        assert_eq!(details.call_type, CallType::Ocr);
        assert_eq!(details.request_id, Some("req_ocr".to_string()));
        assert_eq!(details.litellm_call_id, Some("call_ocr".to_string()));
    }
}
