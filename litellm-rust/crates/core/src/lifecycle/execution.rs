use std::future::Future;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Value, json};

use crate::Error;
use crate::integrations::custom_logger::{CallbackTiming, LogFuture};
use crate::integrations::types::{StandardLoggingMetadata, Usage};

use super::terminal::CostInputs;
use super::{
    ActionResult, ExecutedCall, RouteProjection, StreamingCall, StreamingObserver, StreamingSource,
    TerminalClassification, TerminalRecord,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CallLifecycleContext {
    pub call_type: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub litellm_call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
    pub usage: Usage,
    pub response_cost: f64,
    pub metadata: StandardLoggingMetadata,
}

impl CallLifecycleContext {
    pub fn new(
        call_type: impl Into<String>,
        model: impl Into<String>,
        custom_llm_provider: impl Into<String>,
        litellm_call_id: impl Into<String>,
    ) -> Self {
        Self {
            call_type: call_type.into(),
            model: model.into(),
            custom_llm_provider: custom_llm_provider.into(),
            litellm_call_id: litellm_call_id.into(),
            trace_id: None,
            attempt: 1,
            usage: Usage::default(),
            response_cost: 0.0,
            metadata: StandardLoggingMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: StandardLoggingMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub(super) fn terminal(
        &self,
        timing: CallbackTiming,
        classification: TerminalClassification,
        value: Value,
    ) -> TerminalRecord {
        TerminalRecord {
            call_id: self.litellm_call_id.clone(),
            trace_id: self.trace_id.clone(),
            attempt: self.attempt,
            call_type: self.call_type.clone(),
            model: self.model.clone(),
            provider: self.custom_llm_provider.clone(),
            timing,
            usage: self.usage,
            cost_inputs: CostInputs {
                response_cost: self.response_cost,
                metadata: self.metadata.clone(),
            },
            classification,
            projection: projection(&self.call_type, value),
        }
    }
}

pub trait CallLifecycleRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext;
}

pub trait PreCallHooks<Request>: Send + Sync {
    type PreCallFuture<'a>: Future<Output = ActionResult<Request, Error>> + Send
    where
        Self: 'a;

    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: Request,
    ) -> Self::PreCallFuture<'a>;
}

pub trait ModerationHooks<Request>: Send + Sync {
    type ModerationFuture<'a>: Future<Output = ActionResult<Request, Error>> + Send
    where
        Self: 'a;

    fn async_moderation_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: Request,
    ) -> Self::ModerationFuture<'a>;
}

pub trait TerminalDispatcher: Send + Sync {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a>;
}

pub trait Clock: Send + Sync {
    fn now(&self) -> f64;
}

#[derive(Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> f64 {
        epoch_seconds()
    }
}

#[derive(Default)]
pub struct CallLifecycle;

impl CallLifecycle {
    pub async fn run_streaming<Request, Services, ProviderCall, ProviderFuture>(
        &self,
        context: CallLifecycleContext,
        request: Request,
        services: Arc<Services>,
        observer: Box<dyn StreamingObserver>,
        provider_call: ProviderCall,
    ) -> Result<StreamingCall, Error>
    where
        Services:
            PreCallHooks<Request> + ModerationHooks<Request> + TerminalDispatcher + Clock + 'static,
        ProviderCall: FnOnce(Request) -> ProviderFuture,
        ProviderFuture: Future<Output = Result<StreamingSource, Error>>,
    {
        let start_time = services.now();
        let request = match services.async_pre_call_hook(&context, request).await {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                let executed = failure(&*services, &*services, &context, error, start_time).await;
                return executed.into_result();
            }
        };
        let provider_request = match services.async_moderation_hook(&context, request).await {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                let executed = failure(&*services, &*services, &context, error, start_time).await;
                return executed.into_result();
            }
        };
        match provider_call(provider_request).await {
            Ok(source) => Ok(StreamingCall::new(
                source, observer, context, start_time, services,
            )),
            Err(error) => failure(&*services, &*services, &context, error, start_time)
                .await
                .into_result(),
        }
    }

    pub async fn run<Request, Resp, Policy, Dispatcher, ClockImpl, ProviderCall, ProviderFuture>(
        &self,
        context: CallLifecycleContext,
        request: Request,
        policy: &Policy,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        provider_call: ProviderCall,
    ) -> ExecutedCall<Resp, Error>
    where
        Resp: Serialize,
        Policy: PreCallHooks<Request> + ModerationHooks<Request>,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock,
        ProviderCall: FnOnce(Request) -> ProviderFuture,
        ProviderFuture: Future<Output = Result<Resp, Error>>,
    {
        self.run_with_usage(
            (context, request),
            policy,
            dispatcher,
            clock,
            provider_call,
            |_| None,
        )
        .await
    }

    pub(crate) async fn run_with_usage<
        Request,
        Resp,
        Policy,
        Dispatcher,
        ClockImpl,
        ProviderCall,
        ProviderFuture,
        ResponseUsage,
    >(
        &self,
        input: (CallLifecycleContext, Request),
        policy: &Policy,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        provider_call: ProviderCall,
        response_usage: ResponseUsage,
    ) -> ExecutedCall<Resp, Error>
    where
        Resp: Serialize,
        Policy: PreCallHooks<Request> + ModerationHooks<Request>,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock,
        ProviderCall: FnOnce(Request) -> ProviderFuture,
        ProviderFuture: Future<Output = Result<Resp, Error>>,
        ResponseUsage: FnOnce(&Resp) -> Option<Usage>,
    {
        let (mut context, request) = input;
        let start_time = clock.now();
        let request = match policy.async_pre_call_hook(&context, request).await {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                return failure(dispatcher, clock, &context, error, start_time).await;
            }
        };
        let provider_request = match policy.async_moderation_hook(&context, request).await {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                return failure(dispatcher, clock, &context, error, start_time).await;
            }
        };
        match provider_call(provider_request).await {
            Ok(response) => {
                if let Some(usage) = response_usage(&response) {
                    context.usage = usage;
                }
                let terminal = context.terminal(
                    CallbackTiming::new(start_time, clock.now()),
                    TerminalClassification::Success,
                    serde_json::to_value(&response).unwrap_or(Value::Null),
                );
                let _ = dispatcher.dispatch(&terminal).await;
                ExecutedCall::Success { response, terminal }
            }
            Err(error) => failure(dispatcher, clock, &context, error, start_time).await,
        }
    }

    pub async fn run_prepared<
        InitialRequest,
        ProviderRequest,
        Response,
        Hooks,
        Dispatcher,
        ClockImpl,
        Prepare,
        PrepareFuture,
        ProviderCall,
        ProviderFuture,
    >(
        &self,
        context: CallLifecycleContext,
        request: InitialRequest,
        hooks: &Hooks,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        prepare: Prepare,
        provider_call: ProviderCall,
    ) -> ExecutedCall<Response, Error>
    where
        Response: Serialize,
        Hooks: PreCallHooks<InitialRequest> + ModerationHooks<ProviderRequest>,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock,
        Prepare: FnOnce(InitialRequest) -> PrepareFuture,
        PrepareFuture: Future<Output = Result<ProviderRequest, Error>>,
        ProviderCall: FnOnce(ProviderRequest) -> ProviderFuture,
        ProviderFuture: Future<Output = Result<Response, Error>>,
    {
        let start_time = clock.now();
        let request = match hooks.async_pre_call_hook(&context, request).await {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                return failure(dispatcher, clock, &context, error, start_time).await;
            }
        };
        let provider_request = match prepare(request).await {
            Ok(request) => request,
            Err(error) => {
                return failure(dispatcher, clock, &context, error, start_time).await;
            }
        };
        let provider_request = match hooks
            .async_moderation_hook(&context, provider_request)
            .await
        {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                return failure(dispatcher, clock, &context, error, start_time).await;
            }
        };
        match provider_call(provider_request).await {
            Ok(response) => {
                let terminal = context.terminal(
                    CallbackTiming::new(start_time, clock.now()),
                    TerminalClassification::Success,
                    serde_json::to_value(&response).unwrap_or(Value::Null),
                );
                let _ = dispatcher.dispatch(&terminal).await;
                ExecutedCall::Success { response, terminal }
            }
            Err(error) => failure(dispatcher, clock, &context, error, start_time).await,
        }
    }
}

async fn failure<R, Policy, ClockImpl>(
    policy: &Policy,
    clock: &ClockImpl,
    context: &CallLifecycleContext,
    error: Error,
    start_time: f64,
) -> ExecutedCall<R, Error>
where
    Policy: TerminalDispatcher,
    ClockImpl: Clock,
{
    let kind = error_kind(&error).to_string();
    let message = error.to_string();
    let terminal = context.terminal(
        CallbackTiming::new(start_time, clock.now()),
        TerminalClassification::Failure {
            kind: kind.clone(),
            message: message.clone(),
        },
        json!({"message": message, "kind": kind}),
    );
    let _ = policy.dispatch(&terminal).await;
    ExecutedCall::Failure { error, terminal }
}

fn projection(call_type: &str, value: Value) -> RouteProjection {
    match call_type {
        "ocr" => RouteProjection::Ocr { value },
        "messages" => RouteProjection::Messages { value },
        "chat_completion" | "acompletion" => RouteProjection::ChatCompletions { value },
        "audio_transcription" => RouteProjection::Audio { value },
        "realtime" => RouteProjection::Realtime { value },
        _ => RouteProjection::ResponsesWs { value },
    }
}

fn error_kind(error: &Error) -> &'static str {
    match error {
        Error::Auth(_) => "AuthError",
        Error::InvalidProvider(_) => "InvalidProvider",
        Error::InvalidRequest(_) => "InvalidRequest",
        Error::InvalidType { .. } => "InvalidType",
        Error::MissingField(_) => "MissingField",
        Error::Http { .. } => "HttpError",
        Error::InvalidResponse(_) => "InvalidResponse",
        Error::Network(_) => "NetworkError",
        Error::Connect(_) => "ConnectError",
        Error::Routing(_) => "RoutingError",
        Error::Unsupported(_) => "UnsupportedRequest",
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
    use std::pin::Pin;
    use std::sync::Mutex;

    use super::*;

    type PolicyFuture<'a, T> = Pin<Box<dyn Future<Output = ActionResult<T, Error>> + Send + 'a>>;

    #[derive(Default)]
    struct RecordingPolicy {
        terminals: Mutex<Vec<TerminalRecord>>,
        reject: bool,
    }

    impl PreCallHooks<String> for RecordingPolicy {
        type PreCallFuture<'a> = PolicyFuture<'a, String>;

        fn async_pre_call_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: String,
        ) -> Self::PreCallFuture<'a> {
            Box::pin(async move {
                if self.reject {
                    ActionResult::Reject(Error::InvalidRequest("blocked".to_string()))
                } else {
                    ActionResult::Replace(format!("{request}:pre"))
                }
            })
        }
    }

    impl ModerationHooks<String> for RecordingPolicy {
        type ModerationFuture<'a> = PolicyFuture<'a, String>;

        fn async_moderation_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: String,
        ) -> Self::ModerationFuture<'a> {
            Box::pin(async move { ActionResult::Replace(format!("{request}:during")) })
        }
    }

    impl TerminalDispatcher for RecordingPolicy {
        fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
            Box::pin(async move {
                self.terminals.lock().unwrap().push(terminal.clone());
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn run_applies_replacements_and_returns_terminal_record() {
        let policy = RecordingPolicy::default();
        let executed = CallLifecycle
            .run(
                CallLifecycleContext::new("ocr", "model", "provider", "call-1"),
                "request".to_string(),
                &policy,
                &policy,
                &SystemClock,
                |request| async move {
                    assert_eq!(request, "request:pre:during");
                    Ok(request)
                },
            )
            .await;

        assert!(matches!(
            executed,
            ExecutedCall::Success { ref terminal, .. } if terminal.call_id == "call-1"
        ));
        assert_eq!(policy.terminals.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn rejection_dispatches_and_retains_failure_record() {
        let policy = RecordingPolicy {
            reject: true,
            ..Default::default()
        };
        let executed = CallLifecycle
            .run(
                CallLifecycleContext::new("ocr", "model", "provider", "call-2"),
                "request".to_string(),
                &policy,
                &policy,
                &SystemClock,
                |request| async move { Ok(request) },
            )
            .await;

        assert!(matches!(
            executed,
            ExecutedCall::Failure {
                error: Error::InvalidRequest(_),
                ref terminal,
            } if terminal.call_id == "call-2"
        ));
        assert!(matches!(
            policy.terminals.lock().unwrap()[0].classification,
            TerminalClassification::Failure { .. }
        ));
    }

    #[derive(Default)]
    struct PreparedHooks {
        events: Mutex<Vec<&'static str>>,
    }

    impl PreCallHooks<String> for PreparedHooks {
        type PreCallFuture<'a> = PolicyFuture<'a, String>;

        fn async_pre_call_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> Self::PreCallFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("pre_call");
                ActionResult::Replace(format!("{request}:checked"))
            })
        }
    }

    impl ModerationHooks<usize> for PreparedHooks {
        type ModerationFuture<'a> = PolicyFuture<'a, usize>;

        fn async_moderation_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: usize,
        ) -> Self::ModerationFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("moderation");
                ActionResult::Continue(request)
            })
        }
    }

    impl TerminalDispatcher for PreparedHooks {
        fn dispatch<'a>(&'a self, _: &'a TerminalRecord) -> LogFuture<'a> {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn preparation_is_separate_from_callback_phases() {
        let hooks = PreparedHooks::default();
        let executed = CallLifecycle
            .run_prepared(
                CallLifecycleContext::new("audio_transcription", "model", "provider", "call-3"),
                "request".to_string(),
                &hooks,
                &hooks,
                &SystemClock,
                |request| {
                    hooks.events.lock().unwrap().push("prepare");
                    std::future::ready(Ok(request.len()))
                },
                |request| {
                    hooks.events.lock().unwrap().push("provider");
                    std::future::ready(Ok(request))
                },
            )
            .await;

        assert!(matches!(
            executed,
            ExecutedCall::Success { response: 15, .. }
        ));
        assert_eq!(
            hooks.events.lock().unwrap().as_slice(),
            ["pre_call", "prepare", "moderation", "provider"]
        );
    }
}
