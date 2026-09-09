//! Rust mirror of Python `CustomGuardrail` entrypoints used by the proxy.
//!
//! This module is intentionally Rust-only: Python/PyO3 adapters are a later
//! layer that should implement this trait rather than changing the runner.

use std::sync::Arc;

use futures_util::stream::{self, StreamExt, TryStreamExt};

pub mod types;

pub use types::{
    GuardrailContext, GuardrailDecision, GuardrailDispatchReport, GuardrailError,
    GuardrailEventHook, GuardrailFuture,
};

pub trait CustomGuardrail: Send + Sync {
    type PreCallRequest: Send + 'static;
    type DuringCallRequest: Send + 'static;

    fn guardrail_name(&self) -> &str;

    fn supported_event_hooks(&self) -> &[GuardrailEventHook];

    /// Python 1:1 name: `async_pre_call_hook(user_api_key_dict, cache, data, call_type)`.
    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        request: Self::PreCallRequest,
    ) -> GuardrailFuture<'a, Self::PreCallRequest> {
        Box::pin(async move { Ok(GuardrailDecision::Allow(request)) })
    }

    /// Python 1:1 name: `async_moderation_hook(data, user_api_key_dict, call_type)`.
    fn async_moderation_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        request: Self::DuringCallRequest,
    ) -> GuardrailFuture<'a, Self::DuringCallRequest> {
        Box::pin(async move { Ok(GuardrailDecision::Allow(request)) })
    }
}

#[derive(Clone)]
pub struct CustomGuardrailRunner<PreCallRequest, DuringCallRequest> {
    guardrails: Vec<
        Arc<
            dyn CustomGuardrail<
                    PreCallRequest = PreCallRequest,
                    DuringCallRequest = DuringCallRequest,
                >,
        >,
    >,
}

impl<PreCallRequest, DuringCallRequest> CustomGuardrailRunner<PreCallRequest, DuringCallRequest>
where
    PreCallRequest: Send + 'static,
    DuringCallRequest: Send + 'static,
{
    pub fn new(
        guardrails: Vec<
            Arc<
                dyn CustomGuardrail<
                        PreCallRequest = PreCallRequest,
                        DuringCallRequest = DuringCallRequest,
                    >,
            >,
        >,
    ) -> Self {
        Self { guardrails }
    }

    pub fn is_empty(&self) -> bool {
        self.guardrails.is_empty()
    }

    pub async fn run_pre_call(
        &self,
        context: &GuardrailContext,
        request: PreCallRequest,
    ) -> Result<(PreCallRequest, GuardrailDispatchReport), GuardrailError> {
        stream::iter(self.guardrails.iter().filter(|guardrail| {
            Self::should_run(guardrail.as_ref(), GuardrailEventHook::PreCall, context)
        }))
        .map(Ok::<_, GuardrailError>)
        .try_fold(
            (request, GuardrailDispatchReport::default()),
            |(request, report), guardrail| async move {
                let request = guardrail
                    .async_pre_call_hook(context, request)
                    .await?
                    .into_request()?;
                Ok((
                    request,
                    GuardrailDispatchReport {
                        invoked: report.invoked + 1,
                    },
                ))
            },
        )
        .await
    }

    pub async fn run_during_call(
        &self,
        context: &GuardrailContext,
        request: DuringCallRequest,
    ) -> Result<(DuringCallRequest, GuardrailDispatchReport), GuardrailError> {
        stream::iter(self.guardrails.iter().filter(|guardrail| {
            Self::should_run(guardrail.as_ref(), GuardrailEventHook::DuringCall, context)
        }))
        .map(Ok::<_, GuardrailError>)
        .try_fold(
            (request, GuardrailDispatchReport::default()),
            |(request, report), guardrail| async move {
                let request = guardrail
                    .async_moderation_hook(context, request)
                    .await?
                    .into_request()?;
                Ok((
                    request,
                    GuardrailDispatchReport {
                        invoked: report.invoked + 1,
                    },
                ))
            },
        )
        .await
    }

    fn should_run(
        guardrail: &dyn CustomGuardrail<PreCallRequest = PreCallRequest, DuringCallRequest = DuringCallRequest>,
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
    use crate::integrations::custom_logger::CallType;
    use std::sync::Mutex;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestPreCallRequest {
        content: String,
        masked: bool,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestDuringCallRequest {
        content: String,
        masked: bool,
    }

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

        fn pre_call_decision(
            &self,
            request: TestPreCallRequest,
        ) -> GuardrailDecision<TestPreCallRequest> {
            match self.decision {
                TestDecision::Allow => GuardrailDecision::Allow(request),
                TestDecision::Mask => GuardrailDecision::Mask(TestPreCallRequest {
                    masked: true,
                    ..request
                }),
                TestDecision::Block => {
                    GuardrailDecision::Block(GuardrailError::blocked("blocked by guardrail"))
                }
            }
        }

        fn during_call_decision(
            &self,
            request: TestDuringCallRequest,
        ) -> GuardrailDecision<TestDuringCallRequest> {
            match self.decision {
                TestDecision::Allow => GuardrailDecision::Allow(request),
                TestDecision::Mask => GuardrailDecision::Mask(TestDuringCallRequest {
                    masked: true,
                    ..request
                }),
                TestDecision::Block => {
                    GuardrailDecision::Block(GuardrailError::blocked("blocked by guardrail"))
                }
            }
        }
    }

    impl CustomGuardrail for RecordingCustomGuardrail {
        type PreCallRequest = TestPreCallRequest;
        type DuringCallRequest = TestDuringCallRequest;

        fn guardrail_name(&self) -> &str {
            &self.name
        }

        fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
            &self.hooks
        }

        fn async_pre_call_hook<'a>(
            &'a self,
            _context: &'a GuardrailContext,
            request: TestPreCallRequest,
        ) -> GuardrailFuture<'a, TestPreCallRequest> {
            Box::pin(async move {
                self.calls.lock().unwrap().push("async_pre_call_hook");
                Ok(self.pre_call_decision(request))
            })
        }

        fn async_moderation_hook<'a>(
            &'a self,
            _context: &'a GuardrailContext,
            request: TestDuringCallRequest,
        ) -> GuardrailFuture<'a, TestDuringCallRequest> {
            Box::pin(async move {
                self.calls.lock().unwrap().push("async_moderation_hook");
                Ok(self.during_call_decision(request))
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
        let request = TestPreCallRequest {
            content: "hello".to_string(),
            masked: false,
        };

        let (result, report) = runner
            .run_pre_call(&context, request)
            .await
            .expect("guardrail allows request");

        assert_eq!(report.invoked, 1);
        assert_eq!(result.content, "hello");
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
        let request = TestDuringCallRequest {
            content: "hello".to_string(),
            masked: false,
        };

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
        let request = TestPreCallRequest {
            content: "secret".to_string(),
            masked: false,
        };

        let (result, report) = runner
            .run_pre_call(&context, request)
            .await
            .expect("mask continues");

        assert_eq!(report.invoked, 1);
        assert!(result.masked);
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
        let result = runner
            .run_pre_call(
                &GuardrailContext::new(CallType::Completion),
                TestPreCallRequest {
                    content: "blocked".to_string(),
                    masked: false,
                },
            )
            .await;

        assert!(result.is_err());
        assert_eq!(blocking_guardrail.calls(), vec!["async_pre_call_hook"]);
        assert_eq!(later_guardrail.calls(), Vec::<&'static str>::new());
    }

    #[tokio::test]
    async fn no_guardrails_fast_path_dispatches_nothing() {
        let runner: CustomGuardrailRunner<TestPreCallRequest, TestDuringCallRequest> =
            CustomGuardrailRunner::new(Vec::new());
        let context = GuardrailContext::new(CallType::Ocr);
        let request = TestPreCallRequest {
            content: "ok".to_string(),
            masked: false,
        };

        let (result, report) = runner
            .run_pre_call(&context, request)
            .await
            .expect("no guardrails allow request");

        assert!(runner.is_empty());
        assert_eq!(report, GuardrailDispatchReport::default());
        assert_eq!(result.content, "ok");
    }
}
