//! Rust mirror of Python `CustomGuardrail` entrypoints used by the proxy.
//!
//! This module is intentionally Rust-only: Python/PyO3 adapters are a later
//! layer that should implement this trait rather than changing the runner.

use std::future::Future;
use std::sync::Arc;

use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};

pub mod types;

pub use types::{
    GuardrailContext, GuardrailDecision, GuardrailDispatchReport, GuardrailError,
    GuardrailEventHook, GuardrailFuture, GuardrailRequest,
};

pub trait CustomGuardrail: Send + Sync {
    fn guardrail_name(&self) -> &str;

    fn supported_event_hooks(&self) -> &[GuardrailEventHook];

    /// Python 1:1 name: `async_pre_call_hook(user_api_key_dict, cache, data, call_type)`.
    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move { Ok(GuardrailDecision::Allow(request)) })
    }

    /// Python 1:1 name: `async_moderation_hook(data, user_api_key_dict, call_type)`.
    fn async_moderation_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move { Ok(GuardrailDecision::Allow(request)) })
    }
}

pub struct CustomGuardrailRunner {
    guardrails: Vec<Arc<dyn CustomGuardrail>>,
}

impl CustomGuardrailRunner {
    pub fn new(guardrails: Vec<Arc<dyn CustomGuardrail>>) -> Self {
        Self { guardrails }
    }

    pub fn is_empty(&self) -> bool {
        self.guardrails.is_empty()
    }

    pub async fn run_pre_call(
        &self,
        context: &GuardrailContext,
        request: GuardrailRequest,
    ) -> Result<(GuardrailRequest, GuardrailDispatchReport), GuardrailError> {
        self.run_hook(GuardrailEventHook::PreCall, context, request)
            .await
    }

    pub async fn run_during_call(
        &self,
        context: &GuardrailContext,
        request: GuardrailRequest,
    ) -> Result<(GuardrailRequest, GuardrailDispatchReport), GuardrailError> {
        self.run_hook(GuardrailEventHook::DuringCall, context, request)
            .await
    }

    pub async fn run_before_provider<F, Fut, T>(
        &self,
        event_hook: GuardrailEventHook,
        context: &GuardrailContext,
        request: GuardrailRequest,
        provider: F,
    ) -> Result<T, GuardrailError>
    where
        F: FnOnce(GuardrailRequest) -> Fut,
        Fut: Future<Output = Result<T, GuardrailError>>,
    {
        let (request, _) = self.run_hook(event_hook, context, request).await?;
        provider(request).await
    }

    pub async fn run_pre_call_with_failure_logging(
        &self,
        context: &GuardrailContext,
        request: GuardrailRequest,
        logger_runner: &CustomLoggerRunner,
        model_call_details: &ModelCallDetails,
        timing: CallbackTiming,
    ) -> Result<(GuardrailRequest, GuardrailDispatchReport), GuardrailError> {
        match self.run_pre_call(context, request).await {
            Ok(result) => Ok(result),
            Err(error) => {
                let failure_details = model_call_details.clone().with_failure_error(LoggingError {
                    message: error.message.clone(),
                    kind: error.kind.clone(),
                });
                let response_obj = CallbackValue::new(
                    "guardrail_error",
                    serde_json::json!({
                        "message": error.message,
                        "kind": error.kind,
                    }),
                );
                logger_runner
                    .async_log_failure_event(&failure_details, Some(&response_obj), timing)
                    .await;
                Err(error)
            }
        }
    }

    async fn run_hook(
        &self,
        event_hook: GuardrailEventHook,
        context: &GuardrailContext,
        mut request: GuardrailRequest,
    ) -> Result<(GuardrailRequest, GuardrailDispatchReport), GuardrailError> {
        if self.guardrails.is_empty() {
            return Ok((request, GuardrailDispatchReport::default()));
        }

        let mut report = GuardrailDispatchReport::default();
        for guardrail in &self.guardrails {
            if !self.should_run(guardrail.as_ref(), event_hook, context) {
                continue;
            }

            report.invoked += 1;
            let decision = match event_hook {
                GuardrailEventHook::PreCall => {
                    guardrail
                        .async_pre_call_hook(context, request.clone())
                        .await?
                }
                GuardrailEventHook::DuringCall => {
                    guardrail
                        .async_moderation_hook(context, request.clone())
                        .await?
                }
            };
            match decision.into_request() {
                Ok(next_request) => request = next_request,
                Err(error) => return Err(error),
            }
        }

        Ok((request, report))
    }

    fn should_run(
        &self,
        guardrail: &dyn CustomGuardrail,
        event_hook: GuardrailEventHook,
        context: &GuardrailContext,
    ) -> bool {
        let supports_hook = guardrail.supported_event_hooks().contains(&event_hook);
        let selected = context.selected_guardrails.is_empty()
            || context
                .selected_guardrails
                .iter()
                .any(|name| name == guardrail.guardrail_name());
        supports_hook && selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::{CallType, CallbackValue, CustomLogger, LogFuture};
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;
    use std::sync::Mutex;

    #[derive(Clone)]
    enum TestDecision {
        Allow,
        Mask,
        Block,
    }

    struct RecordingCustomGuardrail {
        name: String,
        hooks: Vec<GuardrailEventHook>,
        decision: TestDecision,
        calls: Mutex<Vec<&'static str>>,
    }

    impl RecordingCustomGuardrail {
        fn new(name: &str, hooks: Vec<GuardrailEventHook>, decision: TestDecision) -> Self {
            Self {
                name: name.to_string(),
                hooks,
                decision,
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<&'static str> {
            self.calls.lock().unwrap().clone()
        }

        fn decision(&self, mut request: GuardrailRequest) -> GuardrailDecision {
            match self.decision {
                TestDecision::Allow => GuardrailDecision::Allow(request),
                TestDecision::Mask => {
                    request.data["masked"] = json!(true);
                    GuardrailDecision::Mask(request)
                }
                TestDecision::Block => {
                    GuardrailDecision::Block(GuardrailError::blocked("blocked by guardrail"))
                }
            }
        }
    }

    impl CustomGuardrail for RecordingCustomGuardrail {
        fn guardrail_name(&self) -> &str {
            &self.name
        }

        fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
            &self.hooks
        }

        fn async_pre_call_hook<'a>(
            &'a self,
            _context: &'a GuardrailContext,
            request: GuardrailRequest,
        ) -> GuardrailFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push("async_pre_call_hook");
                Ok(self.decision(request))
            })
        }

        fn async_moderation_hook<'a>(
            &'a self,
            _context: &'a GuardrailContext,
            request: GuardrailRequest,
        ) -> GuardrailFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push("async_moderation_hook");
                Ok(self.decision(request))
            })
        }
    }

    #[tokio::test]
    async fn pre_call_dispatches_to_async_pre_call_hook() {
        let guardrail = Arc::new(RecordingCustomGuardrail::new(
            "pre",
            vec![GuardrailEventHook::PreCall],
            TestDecision::Allow,
        ));
        let runner = CustomGuardrailRunner::new(vec![guardrail.clone()]);
        let context =
            GuardrailContext::new(CallType::Ocr).with_selected_guardrails(vec!["pre".to_string()]);
        let request = GuardrailRequest::new(json!({"messages": ["hello"]}));

        let (result, report) = runner
            .run_pre_call(&context, request)
            .await
            .expect("guardrail allows request");

        assert_eq!(report.invoked, 1);
        assert_eq!(result.data["messages"], json!(["hello"]));
        assert_eq!(guardrail.calls(), vec!["async_pre_call_hook"]);
    }

    #[tokio::test]
    async fn during_call_dispatches_to_async_moderation_hook() {
        let guardrail = Arc::new(RecordingCustomGuardrail::new(
            "during",
            vec![GuardrailEventHook::DuringCall],
            TestDecision::Allow,
        ));
        let runner = CustomGuardrailRunner::new(vec![guardrail.clone()]);
        let context = GuardrailContext::new(CallType::Completion)
            .with_selected_guardrails(vec!["during".to_string()]);
        let request = GuardrailRequest::new(json!({"prompt": "hello"}));

        let (_result, report) = runner
            .run_during_call(&context, request)
            .await
            .expect("guardrail allows request");

        assert_eq!(report.invoked, 1);
        assert_eq!(guardrail.calls(), vec!["async_moderation_hook"]);
    }

    #[tokio::test]
    async fn mask_decision_continues_with_updated_request() {
        let guardrail = Arc::new(RecordingCustomGuardrail::new(
            "masker",
            vec![GuardrailEventHook::PreCall],
            TestDecision::Mask,
        ));
        let runner = CustomGuardrailRunner::new(vec![guardrail]);
        let context = GuardrailContext::new(CallType::Ocr);
        let request = GuardrailRequest::new(json!({"document": "secret"}));

        let (result, report) = runner
            .run_pre_call(&context, request)
            .await
            .expect("mask continues");

        assert_eq!(report.invoked, 1);
        assert_eq!(result.data["masked"], json!(true));
    }

    #[tokio::test]
    async fn block_decision_short_circuits_and_logs_failure() {
        struct RecordingFailureLogger {
            errors: Mutex<Vec<String>>,
        }

        impl CustomLogger for RecordingFailureLogger {
            fn async_log_failure_event<'a>(
                &'a self,
                model_call_details: &'a ModelCallDetails,
                _response_obj: Option<&'a CallbackValue>,
                _timing: CallbackTiming,
            ) -> LogFuture<'a> {
                Box::pin(async move {
                    self.errors.lock().unwrap().push(
                        model_call_details
                            .failure_error
                            .as_ref()
                            .map(|error| error.kind.clone())
                            .unwrap_or_default(),
                    );
                    Ok(())
                })
            }
        }

        let guardrail = Arc::new(RecordingCustomGuardrail::new(
            "blocker",
            vec![GuardrailEventHook::PreCall],
            TestDecision::Block,
        ));
        let guardrail_runner = CustomGuardrailRunner::new(vec![guardrail]);
        let logger = Arc::new(RecordingFailureLogger {
            errors: Mutex::new(Vec::new()),
        });
        let logger_runner = CustomLoggerRunner::new(vec![logger.clone()]);
        let context = GuardrailContext::new(CallType::Ocr);
        let details = ModelCallDetails::from_standard_logging_payload(StandardLoggingPayload {
            id: "req_ocr".to_string(),
            litellm_call_id: "req_ocr".to_string(),
            call_type: "ocr".to_string(),
            model: "mistral-ocr-latest".to_string(),
            custom_llm_provider: "mistral".to_string(),
            response_cost: 0.0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            start_time: 1.0,
            end_time: 1.0,
            stream: false,
            metadata: StandardLoggingMetadata::default(),
            messages: None,
        });

        let err = guardrail_runner
            .run_pre_call_with_failure_logging(
                &context,
                GuardrailRequest::new(json!({"document": "bad"})),
                &logger_runner,
                &details,
                CallbackTiming::new(1.0, 2.0),
            )
            .await
            .expect_err("guardrail blocks request");

        assert_eq!(err.kind, "GuardrailBlocked");
        assert_eq!(
            logger.errors.lock().unwrap().as_slice(),
            ["GuardrailBlocked"]
        );
    }

    #[tokio::test]
    async fn block_decision_short_circuits_later_guardrails_and_provider_work() {
        let blocking_guardrail = Arc::new(RecordingCustomGuardrail::new(
            "blocker",
            vec![GuardrailEventHook::PreCall],
            TestDecision::Block,
        ));
        let later_guardrail = Arc::new(RecordingCustomGuardrail::new(
            "later",
            vec![GuardrailEventHook::PreCall],
            TestDecision::Allow,
        ));
        let runner =
            CustomGuardrailRunner::new(vec![blocking_guardrail.clone(), later_guardrail.clone()]);
        let provider_called = Arc::new(Mutex::new(false));
        let provider_called_for_closure = provider_called.clone();

        let result = runner
            .run_before_provider(
                GuardrailEventHook::PreCall,
                &GuardrailContext::new(CallType::Completion),
                GuardrailRequest::new(json!({"prompt": "blocked"})),
                move |_request| async move {
                    *provider_called_for_closure.lock().unwrap() = true;
                    Ok("provider response")
                },
            )
            .await;

        assert!(result.is_err());
        assert_eq!(blocking_guardrail.calls(), vec!["async_pre_call_hook"]);
        assert_eq!(later_guardrail.calls(), Vec::<&'static str>::new());
        assert!(!*provider_called.lock().unwrap());
    }

    #[tokio::test]
    async fn run_before_provider_returns_provider_guardrail_error_directly() {
        let guardrail = Arc::new(RecordingCustomGuardrail::new(
            "allow",
            vec![GuardrailEventHook::PreCall],
            TestDecision::Allow,
        ));
        let runner = CustomGuardrailRunner::new(vec![guardrail]);

        let result = runner
            .run_before_provider(
                GuardrailEventHook::PreCall,
                &GuardrailContext::new(CallType::Completion),
                GuardrailRequest::new(json!({"prompt": "allowed"})),
                |_request| async move {
                    Err::<&'static str, GuardrailError>(GuardrailError::blocked(
                        "provider-side guardrail error",
                    ))
                },
            )
            .await;

        let err = result.expect_err("provider error is returned directly");
        assert_eq!(err.kind, "GuardrailBlocked");
        assert_eq!(err.message, "provider-side guardrail error");
    }

    #[tokio::test]
    async fn no_guardrails_fast_path_dispatches_nothing() {
        let runner = CustomGuardrailRunner::new(Vec::new());
        let context = GuardrailContext::new(CallType::Ocr);
        let request = GuardrailRequest::new(json!({"document": "ok"}));

        let (result, report) = runner
            .run_pre_call(&context, request)
            .await
            .expect("no guardrails allow request");

        assert!(runner.is_empty());
        assert_eq!(report, GuardrailDispatchReport::default());
        assert_eq!(result.data["document"], json!("ok"));
    }
}
