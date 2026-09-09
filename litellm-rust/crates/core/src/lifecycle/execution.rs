use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Value, json};

use crate::Error;
use crate::integrations::custom_logger::{CallbackTiming, LogFuture};
use crate::integrations::types::{StandardLoggingMetadata, Usage};

use super::Outcome;
use super::program::{Observations, Operation};
use super::terminal::CostInputs;
use super::{
    ActionResult, CallLifecycle, ExecutedCall, RouteProjection, StreamingCall, StreamingObserver,
    StreamingSource, TerminalClassification, TerminalRecord,
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
    pub provider_usage: super::terminal::UsageObservation,
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
            provider_usage: super::terminal::UsageObservation::Unavailable,
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
            provider_usage: self.provider_usage,
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
        _context: &'a CallLifecycleContext,
        request: Request,
    ) -> Self::PreCallFuture<'a>;
}

pub trait ModerationHooks<Request>: Send + Sync {
    type ModerationFuture<'a>: Future<Output = ActionResult<Request, Error>> + Send
    where
        Self: 'a;

    fn async_moderation_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: Request,
    ) -> Self::ModerationFuture<'a>;
}

pub type CallbackFuture<'a, Output> = Pin<Box<dyn Future<Output = Output> + Send + 'a>>;

pub trait DeploymentPreHooks<Request: Send>: Send + Sync {
    fn async_pre_call_deployment_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: Request,
    ) -> CallbackFuture<'a, ActionResult<Request, Error>>
    where
        Request: 'a,
    {
        Box::pin(async move { ActionResult::Continue(request) })
    }
}

pub trait DeploymentSuccessHooks<Response: Send>: Send + Sync {
    fn async_post_call_success_deployment_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        response: Response,
    ) -> CallbackFuture<'a, ActionResult<Response, Error>>
    where
        Response: 'a,
    {
        Box::pin(async move { ActionResult::Continue(response) })
    }
}

pub trait DeploymentFailureHooks: Send + Sync {
    fn async_post_call_failure_deployment_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a Error,
    ) -> CallbackFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }
}

pub trait TerminalDispatcher: Send + Sync {
    fn deferred_recorder(&self) -> Option<Arc<dyn super::TerminalRecorder>> {
        None
    }

    fn record(&self, terminal: &TerminalRecord) {
        tracing::debug!(target: "litellm::lifecycle", call_id = %terminal.call_id,
            attempt = terminal.attempt, outcome = terminal.classification.kind(), "call completed");
    }

    fn observations(&self) -> Observations {
        Observations {
            logger_available: true,
            has_fallbacks: false,
        }
    }

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

impl CallLifecycle {
    pub async fn run_streaming<Request, Services, ProviderCall, ProviderFuture>(
        self,
        context: CallLifecycleContext,
        request: Request,
        services: Arc<Services>,
        observer: Box<dyn StreamingObserver>,
        provider_call: ProviderCall,
    ) -> Result<StreamingCall, Error>
    where
        Request: Send,
        Services: PreCallHooks<Request>
            + ModerationHooks<Request>
            + DeploymentPreHooks<Request>
            + DeploymentFailureHooks
            + TerminalDispatcher
            + Clock
            + super::StreamDrain
            + 'static,
        ProviderCall: FnOnce(Request) -> ProviderFuture + Send,
        ProviderFuture: Future<Output = Result<StreamingSource, Error>> + Send,
    {
        self.run_streaming_prepared(
            context,
            request,
            services,
            observer,
            |request| std::future::ready(Ok(request)),
            provider_call,
        )
        .await
    }

    pub async fn run_streaming_prepared<
        Request,
        Prepared,
        Services,
        Prepare,
        PrepareFuture,
        ProviderCall,
        ProviderFuture,
    >(
        mut self,
        context: CallLifecycleContext,
        request: Request,
        services: Arc<Services>,
        observer: Box<dyn StreamingObserver>,
        prepare: Prepare,
        provider_call: ProviderCall,
    ) -> Result<StreamingCall, Error>
    where
        Request: Send,
        Services: PreCallHooks<Request>
            + ModerationHooks<Prepared>
            + DeploymentPreHooks<Request>
            + DeploymentFailureHooks
            + TerminalDispatcher
            + Clock
            + super::StreamDrain
            + 'static,
        Prepared: Send,
        Prepare: FnOnce(Request) -> PrepareFuture + Send,
        PrepareFuture: Future<Output = Result<Prepared, Error>> + Send,
        ProviderCall: FnOnce(Prepared) -> ProviderFuture + Send,
        ProviderFuture: Future<Output = Result<StreamingSource, Error>> + Send,
    {
        let start_time = services.now();
        let completion_services = services.clone();
        let mut completion = super::completion::CompletionOwner::new(
            context.clone(),
            start_time,
            &*completion_services,
            &*completion_services,
        );
        let prepared = self
            .prepare_request(
                &context,
                request,
                &*services,
                services.observations(),
                prepare,
            )
            .await;
        let result = match prepared {
            Ok(request) => {
                self.begin_provider();
                let result = provider_call(request).await;
                if result.is_err() {
                    self.advance(Outcome::Failure, services.observations());
                }
                result
            }
            Err(error) => Err(error),
        };
        match result {
            Ok(source) => {
                completion.transfer();
                self.transfer_stream();
                Ok(StreamingCall::with_lifecycle(
                    source, observer, context, start_time, services, self,
                ))
            }
            Err(error) => {
                let terminal = failure_record(&context, &error, start_time, services.now());
                completion.finish(&terminal);
                self.finish_stream(&*services, &context, &terminal, Some((&*services, &error)))
                    .await;
                Err(error)
            }
        }
    }

    #[allow(clippy::manual_async_fn)] // Explicit Send avoids higher-ranked lifetime inference in host futures.
    fn prepare_request<'a, Request, Prepared, Hooks, Prepare, PrepareFuture>(
        &'a mut self,
        context: &'a CallLifecycleContext,
        request: Request,
        hooks: &'a Hooks,
        observations: Observations,
        prepare: Prepare,
    ) -> impl Future<Output = Result<Prepared, Error>> + Send + 'a
    where
        Request: Send + 'a,
        Hooks: PreCallHooks<Request> + DeploymentPreHooks<Request> + ModerationHooks<Prepared>,
        Prepared: Send + 'a,
        Prepare: FnOnce(Request) -> PrepareFuture + Send + 'a,
        PrepareFuture: Future<Output = Result<Prepared, Error>> + Send + 'a,
    {
        async move {
            let mut request = Some(request);
            let mut prepared = None;
            let mut prepare = Some(prepare);
            loop {
                let contract = self
                    .operation()
                    .contract(super::CallbackRuntime::Native, true);
                let result = match self.operation() {
                    Operation::Setup => Ok(()),
                    Operation::DeploymentPre => contract
                        .apply(
                            hooks
                                .async_pre_call_deployment_hook(
                                    context,
                                    request.take().expect("input request"),
                                )
                                .await,
                        )
                        .map(|value| request = Some(value)),
                    Operation::InputHooks => contract
                        .apply(
                            hooks
                                .async_pre_call_hook(
                                    context,
                                    request.take().expect("input request"),
                                )
                                .await,
                        )
                        .map(|value| request = Some(value)),
                    Operation::BuildRequest => prepare.take().expect("prepare once")(
                        request.take().expect("input request"),
                    )
                    .await
                    .map(|value| prepared = Some(value)),
                    Operation::PreCall => contract
                        .apply(
                            hooks
                                .async_moderation_hook(
                                    context,
                                    prepared.take().expect("prepared request"),
                                )
                                .await,
                        )
                        .map(|value| prepared = Some(value)),
                    Operation::Send => return Ok(prepared.take().expect("prepared request")),
                    operation => Err(Error::InvalidRequest(format!(
                        "unexpected preparation operation: {operation:?}"
                    ))),
                };
                self.advance(
                    if result.is_ok() {
                        Outcome::Success
                    } else {
                        Outcome::Failure
                    },
                    observations,
                );
                result?;
            }
        }
    }

    pub async fn run<Request, Resp, Policy, Dispatcher, ClockImpl, ProviderCall, ProviderFuture>(
        self,
        context: CallLifecycleContext,
        request: Request,
        policy: &Policy,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        provider_call: ProviderCall,
    ) -> ExecutedCall<Resp, Error>
    where
        Request: Send,
        Resp: Serialize + Send,
        Policy: PreCallHooks<Request>
            + ModerationHooks<Request>
            + DeploymentPreHooks<Request>
            + DeploymentSuccessHooks<Resp>
            + DeploymentFailureHooks,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock + Sync,
        ProviderCall: FnOnce(Request) -> ProviderFuture + Send,
        ProviderFuture: Future<Output = Result<Resp, Error>> + Send,
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
        self,
        input: (CallLifecycleContext, Request),
        policy: &Policy,
        dispatcher: &Dispatcher,
        clock: &ClockImpl,
        provider_call: ProviderCall,
        response_usage: ResponseUsage,
    ) -> ExecutedCall<Resp, Error>
    where
        Request: Send,
        Resp: Serialize + Send,
        Policy: PreCallHooks<Request>
            + ModerationHooks<Request>
            + DeploymentPreHooks<Request>
            + DeploymentSuccessHooks<Resp>
            + DeploymentFailureHooks,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock + Sync,
        ProviderCall: FnOnce(Request) -> ProviderFuture + Send,
        ProviderFuture: Future<Output = Result<Resp, Error>> + Send,
        ResponseUsage: FnOnce(&Resp) -> Option<Usage> + Send,
    {
        self.run_prepared_with_usage(
            input,
            (policy, dispatcher, clock),
            |request| std::future::ready(Ok(request)),
            provider_call,
            response_usage,
        )
        .await
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
        self,
        context: CallLifecycleContext,
        request: InitialRequest,
        services: (&Hooks, &Dispatcher, &ClockImpl),
        prepare: Prepare,
        provider_call: ProviderCall,
    ) -> ExecutedCall<Response, Error>
    where
        InitialRequest: Send,
        Response: Serialize + Send,
        Hooks: PreCallHooks<InitialRequest>
            + ModerationHooks<ProviderRequest>
            + DeploymentPreHooks<InitialRequest>
            + DeploymentSuccessHooks<Response>
            + DeploymentFailureHooks,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock + Sync,
        ProviderRequest: Send,
        Prepare: FnOnce(InitialRequest) -> PrepareFuture + Send,
        PrepareFuture: Future<Output = Result<ProviderRequest, Error>> + Send,
        ProviderCall: FnOnce(ProviderRequest) -> ProviderFuture + Send,
        ProviderFuture: Future<Output = Result<Response, Error>> + Send,
    {
        self.run_prepared_with_usage((context, request), services, prepare, provider_call, |_| {
            None
        })
        .await
    }

    pub(crate) async fn run_prepared_with_usage<
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
        ResponseUsage,
    >(
        self,
        input: (CallLifecycleContext, InitialRequest),
        services: (&Hooks, &Dispatcher, &ClockImpl),
        prepare: Prepare,
        provider_call: ProviderCall,
        response_usage: ResponseUsage,
    ) -> ExecutedCall<Response, Error>
    where
        InitialRequest: Send,
        Response: Serialize + Send,
        Hooks: PreCallHooks<InitialRequest>
            + ModerationHooks<ProviderRequest>
            + DeploymentPreHooks<InitialRequest>
            + DeploymentSuccessHooks<Response>
            + DeploymentFailureHooks,
        Dispatcher: TerminalDispatcher,
        ClockImpl: Clock + Sync,
        ProviderRequest: Send,
        Prepare: FnOnce(InitialRequest) -> PrepareFuture + Send,
        PrepareFuture: Future<Output = Result<ProviderRequest, Error>> + Send,
        ProviderCall: FnOnce(ProviderRequest) -> ProviderFuture + Send,
        ProviderFuture: Future<Output = Result<Response, Error>> + Send,
        ResponseUsage: FnOnce(&Response) -> Option<Usage> + Send,
    {
        let (hooks, dispatcher, clock) = services;
        let (mut context, request) = input;
        let mut program = self;
        let start_time = clock.now();
        let mut completion =
            super::completion::CompletionOwner::new(context.clone(), start_time, dispatcher, clock);
        let prepared = program
            .prepare_request(&context, request, hooks, dispatcher.observations(), prepare)
            .await;
        let mut result = match prepared {
            Ok(request) => {
                program.begin_provider();
                let result = provider_call(request).await;
                program.advance(
                    if result.is_ok() {
                        Outcome::Success
                    } else {
                        Outcome::Failure
                    },
                    dispatcher.observations(),
                );
                result
            }
            Err(error) => Err(error),
        };
        if let Ok(response) = &result
            && let Some(usage) = response_usage(response)
        {
            context.usage = usage;
            context.provider_usage = super::terminal::UsageObservation::Final(usage);
        }
        completion.update(&context);
        let mut terminal = None;
        loop {
            let outcome = match program.operation() {
                Operation::DeploymentSuccess => {
                    result = Operation::DeploymentSuccess
                        .contract(super::CallbackRuntime::Native, true)
                        .apply(
                            hooks
                                .async_post_call_success_deployment_hook(
                                    &context,
                                    result.expect("provider response"),
                                )
                                .await,
                        );
                    if result.is_ok() {
                        Outcome::Success
                    } else {
                        Outcome::Failure
                    }
                }
                Operation::DeploymentFailure => {
                    let _ = hooks
                        .async_post_call_failure_deployment_hook(
                            &context,
                            result.as_ref().err().expect("call error"),
                        )
                        .await;
                    Outcome::Success
                }
                operation => {
                    if terminal.is_none() {
                        terminal = Some(match &result {
                            Ok(response) => context.terminal(
                                CallbackTiming::new(start_time, clock.now()),
                                TerminalClassification::Success,
                                serde_json::to_value(response).unwrap_or(Value::Null),
                            ),
                            Err(error) => failure_record(&context, error, start_time, clock.now()),
                        });
                    }
                    if result.is_ok()
                        && let Some(recorder) = dispatcher.deferred_recorder()
                    {
                        completion.transfer();
                        return ExecutedCall::Deferred {
                            response: result.expect("successful deferred response"),
                            completion: super::PendingCompletion::new(
                                terminal.expect("terminal record"),
                                recorder,
                            ),
                        };
                    }
                    completion.finish(terminal.as_ref().expect("terminal record"));
                    if let Operation::Complete(_) = operation {
                        let terminal = terminal.expect("terminal record");
                        return match result {
                            Ok(response) => ExecutedCall::Success { response, terminal },
                            Err(error) => ExecutedCall::Failure { error, terminal },
                        };
                    }
                    program
                        .dispatch_terminal(dispatcher, terminal.as_ref().expect("terminal record"))
                        .await;
                    Outcome::Success
                }
            };
            program.advance(outcome, dispatcher.observations());
        }
    }

    async fn dispatch_terminal<D: TerminalDispatcher + ?Sized>(
        &self,
        dispatcher: &D,
        terminal: &TerminalRecord,
    ) {
        if self.delivers_terminal(dispatcher.observations()) {
            super::completion::dispatch(dispatcher, terminal).await;
        }
    }

    pub(super) async fn finish_stream<S: TerminalDispatcher + ?Sized>(
        &mut self,
        services: &S,
        context: &CallLifecycleContext,
        terminal: &TerminalRecord,
        failure: Option<(&dyn DeploymentFailureHooks, &Error)>,
    ) {
        while !matches!(self.operation(), Operation::Complete(_)) {
            if self.operation() == Operation::DeploymentFailure {
                if let Some((hooks, error)) = failure {
                    let _ = hooks
                        .async_post_call_failure_deployment_hook(context, error)
                        .await;
                }
            } else {
                self.dispatch_terminal(services, terminal).await;
            }
            self.advance(Outcome::Success, services.observations());
        }
    }
}

pub(crate) fn provider_result<R: Serialize>(
    context: CallLifecycleContext,
    start_time: f64,
    result: Result<R, Error>,
) -> ExecutedCall<R, Error> {
    match result {
        Ok(response) => {
            let terminal = context.terminal(
                CallbackTiming::new(start_time, SystemClock.now()),
                TerminalClassification::Success,
                serde_json::to_value(&response).unwrap_or(Value::Null),
            );
            ExecutedCall::Success { response, terminal }
        }
        Err(error) => {
            let terminal = failure_record(&context, &error, start_time, SystemClock.now());
            ExecutedCall::Failure { error, terminal }
        }
    }
}

fn failure_record(
    context: &CallLifecycleContext,
    error: &Error,
    start_time: f64,
    end_time: f64,
) -> TerminalRecord {
    let kind = error_kind(error).to_string();
    let message = error.to_string();
    context.terminal(
        CallbackTiming::new(start_time, end_time),
        TerminalClassification::Failure {
            kind: kind.clone(),
            message: message.clone(),
        },
        json!({"message": message, "kind": kind}),
    )
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
        deferred: Option<Arc<RecordedTerminals>>,
        records: Mutex<Vec<TerminalRecord>>,
        terminals: Mutex<Vec<TerminalRecord>>,
        reject: bool,
        reject_response: bool,
        suspend_response: bool,
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

    impl DeploymentPreHooks<String> for RecordingPolicy {}
    impl DeploymentSuccessHooks<String> for RecordingPolicy {
        fn async_post_call_success_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            response: String,
        ) -> CallbackFuture<'a, ActionResult<String, Error>>
        where
            String: 'a,
        {
            Box::pin(async move {
                if self.suspend_response {
                    std::future::pending::<()>().await;
                }
                if self.reject_response {
                    ActionResult::Reject(Error::InvalidRequest("post-response rejection".into()))
                } else {
                    ActionResult::Continue(response)
                }
            })
        }
    }
    impl DeploymentFailureHooks for RecordingPolicy {}

    #[derive(Default)]
    struct RecordedTerminals(Mutex<Vec<TerminalRecord>>);

    impl super::super::TerminalRecorder for RecordedTerminals {
        fn record(&self, terminal: &TerminalRecord) {
            self.0.lock().unwrap().push(terminal.clone());
        }
    }

    impl TerminalDispatcher for RecordingPolicy {
        fn deferred_recorder(&self) -> Option<Arc<dyn super::super::TerminalRecorder>> {
            self.deferred
                .clone()
                .map(|recorder| recorder as Arc<dyn super::super::TerminalRecorder>)
        }
        fn record(&self, terminal: &TerminalRecord) {
            self.records.lock().unwrap().push(terminal.clone());
        }
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
        let executed = CallLifecycle::asynchronous()
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
        let executed = CallLifecycle::asynchronous()
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

    #[tokio::test]
    async fn dropping_provider_future_records_cancellation_without_callback_dispatch() {
        let policy = RecordingPolicy::default();
        let mut call = Box::pin(CallLifecycle::asynchronous().run(
            CallLifecycleContext::new("messages", "model", "provider", "cancelled-call"),
            "request".to_string(),
            &policy,
            &policy,
            &SystemClock,
            |_| std::future::pending::<Result<String, Error>>(),
        ));
        assert!(futures_util::poll!(&mut call).is_pending());
        drop(call);
        assert!(policy.terminals.lock().unwrap().is_empty());
        let records = policy.records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert!(matches!(
            records[0].classification,
            TerminalClassification::Cancelled { .. }
        ));
    }

    #[tokio::test]
    async fn deferred_completion_records_only_acceptance_rejection_or_abandonment() {
        for decision in ["accept", "reject", "drop", "into_result"] {
            let recorder = Arc::new(RecordedTerminals::default());
            let policy = RecordingPolicy {
                deferred: Some(recorder.clone()),
                ..RecordingPolicy::default()
            };
            let call = CallLifecycle::asynchronous()
                .run(
                    CallLifecycleContext::new("ocr", "model", "provider", decision),
                    "request".to_string(),
                    &policy,
                    &policy,
                    &SystemClock,
                    |request| std::future::ready(Ok(request)),
                )
                .await;
            let ExecutedCall::Deferred {
                response,
                completion,
            } = call
            else {
                panic!("expected deferred completion")
            };
            assert_eq!(response, "request:pre:during");
            assert!(recorder.0.lock().unwrap().is_empty());
            assert!(policy.terminals.lock().unwrap().is_empty());
            assert!(policy.records.lock().unwrap().is_empty());
            match decision {
                "accept" => {
                    completion.accept();
                }
                "reject" => {
                    completion.reject("GuardrailBlocked".into(), "blocked".into());
                }
                "into_result" => {
                    assert_eq!(
                        ExecutedCall::<_, Error>::Deferred {
                            response,
                            completion
                        }
                        .into_result()
                        .unwrap(),
                        "request:pre:during"
                    );
                }
                _ => drop(completion),
            }
            let records = recorder.0.lock().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(
                records[0].classification.kind(),
                match decision {
                    "accept" | "into_result" => "Success",
                    "reject" => "GuardrailBlocked",
                    _ => "Cancelled",
                }
            );
        }
    }

    #[tokio::test]
    async fn response_replacement_cannot_rewrite_provider_usage() {
        let hooks = DeploymentRecordingHooks::default();
        let call = CallLifecycle::asynchronous()
            .run_with_usage(
                (
                    CallLifecycleContext::new("messages", "model", "provider", "usage-call"),
                    "request".to_string(),
                ),
                &hooks,
                &hooks,
                &SystemClock,
                |_| std::future::ready(Ok("original".to_string())),
                |response| {
                    Some(Usage {
                        prompt_tokens: 0,
                        completion_tokens: response.len() as u64,
                        total_tokens: response.len() as u64,
                    })
                },
            )
            .await;
        assert_eq!(call.terminal().usage.total_tokens, 8);
        assert_eq!(call.into_result().unwrap(), "original:post");
    }

    #[tokio::test]
    async fn rejection_and_cancellation_after_response_preserve_trusted_usage() {
        for cancelled in [false, true] {
            let policy = RecordingPolicy {
                reject_response: !cancelled,
                suspend_response: cancelled,
                ..Default::default()
            };
            let mut pending = Box::pin(CallLifecycle::asynchronous().run_with_usage(
                (
                    CallLifecycleContext::new("messages", "model", "provider", "usage"),
                    "input".to_string(),
                ),
                &policy,
                &policy,
                &SystemClock,
                |_| std::future::ready(Ok("provider-response".to_string())),
                |_| {
                    Some(Usage {
                        prompt_tokens: 5,
                        completion_tokens: 7,
                        total_tokens: 12,
                    })
                },
            ));
            if cancelled {
                assert!(futures_util::poll!(&mut pending).is_pending());
                assert!(policy.records.lock().unwrap().is_empty());
                drop(pending);
            } else {
                assert!(
                    matches!(pending.await, ExecutedCall::Failure { error: Error::InvalidRequest(message), .. } if message == "post-response rejection")
                );
            }
            let records = policy.records.lock().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].usage.total_tokens, 12);
            assert_eq!(
                records[0].provider_usage,
                super::super::terminal::UsageObservation::Final(records[0].usage)
            );
            assert_eq!(
                records[0].classification.kind(),
                if cancelled {
                    "Cancelled"
                } else {
                    "InvalidRequest"
                }
            );
            assert_eq!(
                policy.terminals.lock().unwrap().len(),
                usize::from(!cancelled)
            );
        }
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

    impl DeploymentPreHooks<String> for PreparedHooks {}
    impl DeploymentSuccessHooks<usize> for PreparedHooks {}
    impl DeploymentFailureHooks for PreparedHooks {}

    impl TerminalDispatcher for PreparedHooks {
        fn dispatch<'a>(&'a self, _: &'a TerminalRecord) -> LogFuture<'a> {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn preparation_is_separate_from_callback_phases() {
        let hooks = PreparedHooks::default();
        let executed = CallLifecycle::asynchronous()
            .run_prepared(
                CallLifecycleContext::new("audio_transcription", "model", "provider", "call-3"),
                "request".to_string(),
                (&hooks, &hooks, &SystemClock),
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

    #[derive(Default)]
    struct DeploymentRecordingHooks {
        events: Mutex<Vec<&'static str>>,
        terminals: Mutex<Vec<TerminalRecord>>,
    }

    impl PreCallHooks<String> for DeploymentRecordingHooks {
        type PreCallFuture<'a> = PolicyFuture<'a, String>;

        fn async_pre_call_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> Self::PreCallFuture<'a> {
            Box::pin(async move { ActionResult::Continue(request) })
        }
    }

    impl ModerationHooks<String> for DeploymentRecordingHooks {
        type ModerationFuture<'a> = PolicyFuture<'a, String>;

        fn async_moderation_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> Self::ModerationFuture<'a> {
            Box::pin(async move { ActionResult::Continue(request) })
        }
    }

    impl DeploymentPreHooks<String> for DeploymentRecordingHooks {
        fn async_pre_call_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> CallbackFuture<'a, ActionResult<String, Error>>
        where
            String: 'a,
        {
            Box::pin(async move {
                self.events.lock().unwrap().push("deployment_pre");
                ActionResult::Replace(format!("{request}:pre"))
            })
        }
    }

    impl DeploymentSuccessHooks<String> for DeploymentRecordingHooks {
        fn async_post_call_success_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            response: String,
        ) -> CallbackFuture<'a, ActionResult<String, Error>>
        where
            String: 'a,
        {
            Box::pin(async move {
                self.events.lock().unwrap().push("deployment_success");
                ActionResult::Replace(format!("{response}:post"))
            })
        }
    }

    impl DeploymentFailureHooks for DeploymentRecordingHooks {
        fn async_post_call_failure_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            _: &'a Error,
        ) -> CallbackFuture<'a, Result<(), Error>> {
            Box::pin(async move {
                self.events.lock().unwrap().push("deployment_failure");
                Err(Error::InvalidRequest("callback failed".to_string()))
            })
        }
    }

    impl TerminalDispatcher for DeploymentRecordingHooks {
        fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("terminal");
                self.terminals.lock().unwrap().push(terminal.clone());
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn deployment_replacements_are_adopted_before_terminal_dispatch() {
        let hooks = DeploymentRecordingHooks::default();
        let executed = CallLifecycle::asynchronous()
            .run(
                CallLifecycleContext::new("chat_completion", "model", "provider", "call-4"),
                "request".to_string(),
                &hooks,
                &hooks,
                &SystemClock,
                |request| std::future::ready(Ok(request)),
            )
            .await;

        assert!(matches!(
            executed,
            ExecutedCall::Success { response, .. } if response == "request:pre:post"
        ));
        assert_eq!(
            hooks.events.lock().unwrap().as_slice(),
            ["deployment_pre", "deployment_success", "terminal"]
        );
    }

    #[tokio::test]
    async fn deployment_failure_hook_cannot_replace_the_original_error() {
        let hooks = DeploymentRecordingHooks::default();
        let executed: ExecutedCall<String, Error> = CallLifecycle::asynchronous()
            .run(
                CallLifecycleContext::new("chat_completion", "model", "provider", "call-5"),
                "request".to_string(),
                &hooks,
                &hooks,
                &SystemClock,
                |_| std::future::ready(Err(Error::Network("provider failed".to_string()))),
            )
            .await;

        assert!(matches!(
            executed,
            ExecutedCall::Failure {
                error: Error::Network(message),
                ..
            } if message == "provider failed"
        ));
        assert_eq!(
            hooks.events.lock().unwrap().as_slice(),
            ["deployment_pre", "deployment_failure", "terminal"]
        );
    }
    struct TraceHooks {
        records: Mutex<Vec<TerminalRecord>>,
        events: Mutex<Vec<Operation>>,
        terminals: Mutex<Vec<TerminalRecord>>,
        reject: Option<Operation>,
        observations: Observations,
    }

    impl TraceHooks {
        fn apply(&self, operation: Operation, value: String) -> ActionResult<String, Error> {
            self.events.lock().unwrap().push(operation);
            if self.reject == Some(operation) {
                ActionResult::Reject(Error::InvalidRequest(format!("{operation:?}")))
            } else {
                ActionResult::Replace(format!("{value}:{operation:?}"))
            }
        }
    }

    impl PreCallHooks<String> for TraceHooks {
        type PreCallFuture<'a> = std::future::Ready<ActionResult<String, Error>>;
        fn async_pre_call_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> Self::PreCallFuture<'a> {
            std::future::ready(self.apply(Operation::InputHooks, request))
        }
    }

    impl ModerationHooks<String> for TraceHooks {
        type ModerationFuture<'a> = std::future::Ready<ActionResult<String, Error>>;
        fn async_moderation_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> Self::ModerationFuture<'a> {
            std::future::ready(self.apply(Operation::PreCall, request))
        }
    }

    impl DeploymentPreHooks<String> for TraceHooks {
        fn async_pre_call_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            request: String,
        ) -> CallbackFuture<'a, ActionResult<String, Error>>
        where
            String: 'a,
        {
            Box::pin(std::future::ready(
                self.apply(Operation::DeploymentPre, request),
            ))
        }
    }

    impl DeploymentSuccessHooks<String> for TraceHooks {
        fn async_post_call_success_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            response: String,
        ) -> CallbackFuture<'a, ActionResult<String, Error>>
        where
            String: 'a,
        {
            Box::pin(std::future::ready(
                self.apply(Operation::DeploymentSuccess, response),
            ))
        }
    }

    impl DeploymentFailureHooks for TraceHooks {
        fn async_post_call_failure_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            _: &'a Error,
        ) -> CallbackFuture<'a, Result<(), Error>> {
            self.events
                .lock()
                .unwrap()
                .push(Operation::DeploymentFailure);
            Box::pin(std::future::ready(Err(Error::Network(
                "observer failed".into(),
            ))))
        }
    }

    impl TerminalDispatcher for TraceHooks {
        fn record(&self, terminal: &TerminalRecord) {
            self.records.lock().unwrap().push(terminal.clone());
        }
        fn observations(&self) -> Observations {
            self.observations
        }
        fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
            self.terminals.lock().unwrap().push(terminal.clone());
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn actual_traces_follow_the_program_across_options_and_failures() {
        use Operation::*;
        for asynchronous in [false, true] {
            for internal_call in [false, true] {
                for logger_available in [false, true] {
                    for has_fallbacks in [false, true] {
                        for reject in [
                            None,
                            Some(DeploymentPre),
                            Some(InputHooks),
                            Some(BuildRequest),
                            Some(PreCall),
                            Some(Send),
                            Some(DeploymentSuccess),
                        ] {
                            if !asynchronous
                                && matches!(reject, Some(DeploymentPre | DeploymentSuccess))
                            {
                                continue;
                            }
                            let observations = Observations {
                                logger_available,
                                has_fallbacks,
                            };
                            let hooks = TraceHooks {
                                records: Mutex::new(Vec::new()),
                                events: Mutex::new(Vec::new()),
                                terminals: Mutex::new(Vec::new()),
                                reject,
                                observations,
                            };
                            let options = super::super::program::ProgramOptions {
                                asynchronous,
                                internal_call,
                            };
                            let executed = CallLifecycle::planned(options)
                                .run_prepared(
                                    CallLifecycleContext::new(
                                        "messages", "model", "provider", "trace",
                                    ),
                                    "input".to_string(),
                                    (&hooks, &hooks, &SystemClock),
                                    |request| {
                                        assert!(request.ends_with(":InputHooks"));
                                        std::future::ready(
                                            hooks.apply(BuildRequest, request).into_result(),
                                        )
                                    },
                                    |request| {
                                        assert!(request.ends_with(":BuildRequest:PreCall"));
                                        std::future::ready(hooks.apply(Send, request).into_result())
                                    },
                                )
                                .await;
                            let mut expected = if asynchronous {
                                vec![
                                    DeploymentPre,
                                    InputHooks,
                                    BuildRequest,
                                    PreCall,
                                    Send,
                                    DeploymentSuccess,
                                ]
                            } else {
                                vec![InputHooks, BuildRequest, PreCall, Send]
                            };
                            if let Some(reject) = reject {
                                expected.truncate(
                                    expected.iter().position(|op| *op == reject).unwrap() + 1,
                                );
                                if asynchronous
                                    && matches!(reject, InputHooks | BuildRequest | PreCall | Send)
                                {
                                    expected.push(DeploymentFailure);
                                }
                                assert!(
                                    matches!(&executed, ExecutedCall::Failure { error: Error::InvalidRequest(message), .. } if *message == format!("{reject:?}"))
                                );
                            } else {
                                let suffix = if asynchronous {
                                    ":BuildRequest:PreCall:Send:DeploymentSuccess"
                                } else {
                                    ":BuildRequest:PreCall:Send"
                                };
                                assert!(
                                    matches!(&executed, ExecutedCall::Success { response, .. } if response.ends_with(suffix))
                                );
                            }
                            assert_eq!(
                                *hooks.events.lock().unwrap(),
                                expected,
                                "{options:?}, {observations:?}, {reject:?}"
                            );
                            let should_dispatch = logger_available
                                && !(reject.is_some() && asynchronous && internal_call);
                            assert_eq!(
                                hooks.terminals.lock().unwrap().len(),
                                usize::from(should_dispatch)
                            );
                            assert_eq!(executed.terminal().call_id, "trace");
                            assert_eq!(hooks.records.lock().unwrap().len(), 1);
                        }
                    }
                }
            }
        }
    }

    impl Clock for TraceHooks {
        fn now(&self) -> f64 {
            10.0
        }
    }

    impl super::super::StreamDrain for TraceHooks {}

    struct TraceObserver;

    impl StreamingObserver for TraceObserver {
        fn observe(&mut self, _: &bytes::Bytes) -> Result<(), Error> {
            Ok(())
        }
        fn usage(&self) -> Usage {
            Usage::default()
        }
        fn projection(&self) -> Value {
            json!({"stream": true})
        }
    }

    #[tokio::test]
    async fn stream_handoff_defers_terminal_and_never_reenters_deployment_hooks() {
        use futures_util::StreamExt;
        for asynchronous in [false, true] {
            for internal_call in [false, true] {
                for stream_error in [false, true] {
                    let services = Arc::new(TraceHooks {
                        records: Mutex::new(Vec::new()),
                        events: Mutex::new(Vec::new()),
                        terminals: Mutex::new(Vec::new()),
                        reject: None,
                        observations: Observations {
                            logger_available: true,
                            has_fallbacks: false,
                        },
                    });
                    let source = StreamingSource {
                        metadata: super::super::StreamingMetadata {
                            status: 200,
                            content_type: None,
                            cache_control: None,
                        },
                        stream: Box::pin(futures_util::stream::iter(if stream_error {
                            vec![Err(Error::Network("stream failed".into()))]
                        } else {
                            vec![Ok(bytes::Bytes::from_static(b"data"))]
                        })),
                    };
                    let mut call = CallLifecycle::planned(super::super::program::ProgramOptions {
                        asynchronous,
                        internal_call,
                    })
                    .run_streaming_prepared(
                        CallLifecycleContext::new("messages", "model", "provider", "stream-trace"),
                        "input".to_string(),
                        services.clone(),
                        Box::new(TraceObserver),
                        |request| {
                            std::future::ready(
                                services
                                    .apply(Operation::BuildRequest, request)
                                    .into_result(),
                            )
                        },
                        |request| {
                            assert!(request.ends_with(":BuildRequest:PreCall"));
                            services.events.lock().unwrap().push(Operation::Send);
                            std::future::ready(Ok(source))
                        },
                    )
                    .await
                    .unwrap();
                    assert!(services.terminals.lock().unwrap().is_empty());
                    assert!(services.records.lock().unwrap().is_empty());
                    let completion = call.completion.register();
                    while call.stream.next().await.is_some() {}
                    let terminal = completion.await.unwrap();
                    assert_eq!(terminal.call_id, "stream-trace");
                    assert_eq!(
                        matches!(
                            terminal.classification,
                            TerminalClassification::Failure { .. }
                        ),
                        stream_error
                    );
                    assert_eq!(
                        services.terminals.lock().unwrap().len(),
                        usize::from(!(stream_error && asynchronous && internal_call))
                    );
                    let mut expected = Vec::new();
                    if asynchronous {
                        expected.push(Operation::DeploymentPre);
                    }
                    expected.extend([
                        Operation::InputHooks,
                        Operation::BuildRequest,
                        Operation::PreCall,
                        Operation::Send,
                    ]);
                    assert_eq!(*services.events.lock().unwrap(), expected);
                }
            }
        }
    }
}
