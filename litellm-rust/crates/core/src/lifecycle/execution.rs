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

pub trait RequestPolicy<InitialReq, ProviderReq>: Send + Sync {
    type PreCallFuture<'a>: Future<Output = ActionResult<InitialReq, Error>> + Send
    where
        Self: 'a;

    type DuringCallFuture<'a>: Future<Output = ActionResult<ProviderReq, Error>> + Send
    where
        Self: 'a;

    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: InitialReq,
    ) -> Self::PreCallFuture<'a>;

    fn async_during_call_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: InitialReq,
    ) -> Self::DuringCallFuture<'a>;
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
    pub async fn run_streaming<InitialReq, ProviderReq, Services, ProviderCall, ProviderFuture>(
        &self,
        context: CallLifecycleContext,
        request: InitialReq,
        services: Arc<Services>,
        observer: Box<dyn StreamingObserver>,
        provider_call: ProviderCall,
    ) -> Result<StreamingCall, Error>
    where
        Services: RequestPolicy<InitialReq, ProviderReq> + TerminalDispatcher + Clock + 'static,
        ProviderCall: FnOnce(ProviderReq) -> ProviderFuture,
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
        let provider_request = match services.async_during_call_hook(&context, request).await {
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

    pub async fn run<
        InitialReq,
        ProviderReq,
        Resp,
        Policy,
        Dispatcher,
        ClockImpl,
        ProviderCall,
        ProviderFuture,
    >(
        &self,
        context: CallLifecycleContext,
        request: InitialReq,
        policy: &Policy,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        provider_call: ProviderCall,
    ) -> ExecutedCall<Resp, Error>
    where
        Resp: Serialize,
        Policy: RequestPolicy<InitialReq, ProviderReq>,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock,
        ProviderCall: FnOnce(ProviderReq) -> ProviderFuture,
        ProviderFuture: Future<Output = Result<Resp, Error>>,
    {
        self.run_with_usage(
            context,
            request,
            policy,
            dispatcher,
            clock,
            provider_call,
            |_| None,
        )
        .await
    }

    pub(crate) async fn run_with_usage<
        InitialReq,
        ProviderReq,
        Resp,
        Policy,
        Dispatcher,
        ClockImpl,
        ProviderCall,
        ProviderFuture,
        ResponseUsage,
    >(
        &self,
        mut context: CallLifecycleContext,
        request: InitialReq,
        policy: &Policy,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        provider_call: ProviderCall,
        response_usage: ResponseUsage,
    ) -> ExecutedCall<Resp, Error>
    where
        Resp: Serialize,
        Policy: RequestPolicy<InitialReq, ProviderReq>,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock,
        ProviderCall: FnOnce(ProviderReq) -> ProviderFuture,
        ProviderFuture: Future<Output = Result<Resp, Error>>,
        ResponseUsage: FnOnce(&Resp) -> Option<Usage>,
    {
        let start_time = clock.now();
        let request = match policy.async_pre_call_hook(&context, request).await {
            ActionResult::Continue(request) | ActionResult::Replace(request) => request,
            ActionResult::Reject(error) => {
                return failure(dispatcher, clock, &context, error, start_time).await;
            }
        };
        let provider_request = match policy.async_during_call_hook(&context, request).await {
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

    impl RequestPolicy<String, String> for RecordingPolicy {
        type PreCallFuture<'a> = PolicyFuture<'a, String>;
        type DuringCallFuture<'a> = PolicyFuture<'a, String>;

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

        fn async_during_call_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: String,
        ) -> Self::DuringCallFuture<'a> {
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
}
