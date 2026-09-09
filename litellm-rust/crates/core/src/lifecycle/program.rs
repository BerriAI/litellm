use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    CallbackRuntime, Delivery, ErrorDisposition, FailurePolicy, OperationContract, Outcome,
    ResultPolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Setup,
    DeploymentPre,
    InputHooks,
    BuildRequest,
    PreCall,
    Send,
    StreamComplete,
    DeploymentSuccess,
    DeploymentFailure,
    SyncSuccess,
    AsyncSuccess,
    SyncSuccessIfNeeded,
    SyncFailure,
    AsyncFailure,
    Restore,
    Complete(Outcome),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Observations {
    pub logger_available: bool,
    pub has_fallbacks: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Commitment {
    BeforeProvider,
    ProviderStarted,
    ResponseReceived,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureStage {
    BeforeProvider,
    ProviderCall,
    AfterProviderResponse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub operation: Operation,
    pub error: ErrorDisposition,
    pub commitment: Commitment,
    pub failure_stage: Option<FailureStage>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProgramOptions {
    pub asynchronous: bool,
    pub internal_call: bool,
}

#[derive(Debug)]
pub struct OperationTicket {
    owner: u64,
    generation: u64,
    operation: Operation,
}

impl OperationTicket {
    pub fn operation(&self) -> Operation {
        self.operation
    }
}

/// ```compile_fail
/// use litellm_core::lifecycle::program::PreparationPermit;
/// use litellm_core::messages::lifecycle::MessagesRoute;
/// let permit: PreparationPermit<MessagesRoute> = PreparationPermit { _route: Default::default() };
/// ```
/// ```compile_fail
/// use litellm_core::lifecycle::CallLifecycle;
/// use litellm_core::messages::lifecycle::MessagesRoute;
/// let mut call = CallLifecycle::asynchronous();
/// let ticket = call.issue().unwrap();
/// let permit = call.preparation_permit::<MessagesRoute>(&ticket);
/// ```
#[derive(Debug)]
pub struct PreparationPermit<Route> {
    _route: std::marker::PhantomData<Route>,
}

impl PreparationPermit<crate::messages::lifecycle::MessagesRoute> {
    pub fn messages(
        self,
        request: crate::messages::types::MessagesOptions,
    ) -> Result<crate::messages::types::MessagesEndpoint, crate::Error> {
        crate::messages::request::build_endpoint(request)
    }
}

impl PreparationPermit<super::ocr::OcrRoute> {
    pub fn ocr(
        self,
        request: crate::ocr::OcrAdmissionRequest,
        auth: &dyn litellm_auth::AuthServices,
    ) -> Result<crate::ocr::OcrPreCallRequest, crate::Error> {
        crate::ocr::request::build_pre_call_request_with_auth(request, auth)
    }
}

impl PreparationPermit<crate::chat_completions::lifecycle::ChatCompletionsRoute> {
    pub async fn chat_completions(
        self,
        services: &impl crate::providers::auth::ChatAuthorizationServices,
        request: crate::chat_completions::types::ChatCompletionsRequest<'_>,
    ) -> Result<crate::chat_completions::types::ChatPreCallRequest, crate::Error> {
        crate::chat_completions::request::build_pre_call_request_with_services(services, request)
            .await
    }

    pub fn chat_request(
        self,
        request: crate::chat_completions::types::ResolvedChatCompletionsRequest<'_>,
    ) -> Result<crate::chat_completions::types::ProviderChatCompletionsRequest, crate::Error> {
        crate::chat_completions::request::build_provider_request(request)
    }
}

/// ```compile_fail
/// use litellm_core::lifecycle::program::ProviderPermit;
/// use litellm_core::messages::lifecycle::MessagesRoute;
/// let permit: ProviderPermit<MessagesRoute> = ProviderPermit { _route: Default::default() };
/// ```
/// ```compile_fail
/// use litellm_core::messages::execute_provider_messages_request;
/// ```
/// ```compile_fail
/// use litellm_core::http_utils::http_request;
/// ```
#[derive(Debug)]
pub struct ProviderPermit<Route> {
    _route: std::marker::PhantomData<Route>,
}

impl ProviderPermit<crate::messages::lifecycle::MessagesRoute> {
    pub async fn messages(
        self,
        request: crate::messages::types::ProviderMessagesRequest,
    ) -> Result<crate::messages::types::AnthropicMessagesResponse, crate::Error> {
        crate::messages::execute_provider_messages_request(request).await
    }

    pub async fn messages_stream<S>(
        self,
        request: crate::messages::types::ProviderMessagesRequest,
        context: super::CallLifecycleContext,
        start_time: f64,
        services: std::sync::Arc<S>,
    ) -> Result<super::StreamingCall, crate::Error>
    where
        S: super::Clock + super::TerminalDispatcher + super::StreamDrain + 'static,
    {
        crate::messages::messages_stream_prepared(request, context, start_time, services).await
    }
}

impl ProviderPermit<crate::chat_completions::lifecycle::ChatCompletionsRoute> {
    pub async fn chat_completions(
        self,
        services: &dyn crate::providers::auth::ChatAuthorizationServices,
        request: crate::chat_completions::types::ChatPreCallRequest,
        readback: crate::chat_completions::types::ChatPreCallReadback,
        context: super::CallLifecycleContext,
    ) -> super::ExecutedCall<crate::chat_completions::types::ChatCompletionsResponse, crate::Error>
    {
        match crate::chat_completions::request::settle_pre_call_request_with_services(
            services, request, readback,
        )
        .await
        {
            Ok(request) => {
                crate::chat_completions::lifecycle::execute_settled(request, context).await
            }
            Err(error) => super::execution::provider_result(
                context,
                super::Clock::now(&super::SystemClock),
                Err(error),
            ),
        }
    }
}

impl ProviderPermit<super::ocr::OcrRoute> {
    pub async fn ocr(
        self,
        request: crate::ocr::SettledOcrRequest,
        context: super::CallLifecycleContext,
    ) -> super::ExecutedCall<serde_json::Value, crate::Error> {
        let start_time = super::Clock::now(&super::SystemClock);
        let result = crate::ocr::send(request)
            .await
            .map(crate::ocr::types::OcrResponseData::into_json);
        super::execution::provider_result(context, start_time, result)
    }
}

#[derive(Debug)]
pub struct CallLifecycle {
    owner: u64,
    generation: u64,
    issued: bool,
    provider_issued: bool,
    preparation_issued: bool,
    operation: Operation,
    outcome: Outcome,
    commitment: Commitment,
    failure_stage: Option<FailureStage>,
    options: ProgramOptions,
}

impl CallLifecycle {
    pub fn planned(options: ProgramOptions) -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            issued: false,
            provider_issued: false,
            preparation_issued: false,
            operation: Operation::Setup,
            outcome: Outcome::Success,
            commitment: Commitment::BeforeProvider,
            failure_stage: None,
            options,
        }
    }

    pub fn asynchronous() -> Self {
        Self::planned(ProgramOptions {
            asynchronous: true,
            internal_call: false,
        })
    }

    pub fn operation(&self) -> Operation {
        self.operation
    }

    pub fn issue(&mut self) -> Result<OperationTicket, crate::Error> {
        if self.issued || matches!(self.operation, Operation::Complete(_)) {
            return Err(crate::Error::InvalidRequest(
                "lifecycle operation is already issued or complete".into(),
            ));
        }
        self.issued = true;
        Ok(OperationTicket {
            owner: self.owner,
            generation: self.generation,
            operation: self.operation,
        })
    }

    fn owns(&self, ticket: &OperationTicket) -> bool {
        self.issued
            && ticket.owner == self.owner
            && ticket.generation == self.generation
            && ticket.operation == self.operation
    }

    pub fn complete_operation(
        &mut self,
        ticket: OperationTicket,
        outcome: Outcome,
        observations: Observations,
    ) -> Result<Transition, crate::Error> {
        if !self.owns(&ticket) {
            return Err(crate::Error::InvalidRequest(
                "stale or foreign lifecycle operation".into(),
            ));
        }
        self.issued = false;
        self.advance(outcome, observations)
            .ok_or_else(|| crate::Error::InvalidRequest("lifecycle is complete".into()))
    }

    pub(crate) fn preparation_permit<Route>(
        &mut self,
        ticket: &OperationTicket,
    ) -> Result<PreparationPermit<Route>, crate::Error> {
        if !self.owns(ticket)
            || ticket.operation != Operation::BuildRequest
            || self.preparation_issued
        {
            return Err(crate::Error::InvalidRequest(
                "request preparation requires the current build operation".into(),
            ));
        }
        self.preparation_issued = true;
        Ok(PreparationPermit {
            _route: std::marker::PhantomData,
        })
    }

    pub(crate) fn provider_permit<Route>(
        &mut self,
        ticket: &OperationTicket,
    ) -> Result<ProviderPermit<Route>, crate::Error> {
        if !self.owns(ticket) || ticket.operation != Operation::Send || self.provider_issued {
            return Err(crate::Error::InvalidRequest(
                "provider execution requires the current send operation".into(),
            ));
        }
        self.provider_issued = true;
        self.begin_provider();
        Ok(ProviderPermit {
            _route: std::marker::PhantomData,
        })
    }

    pub(crate) fn begin_provider(&mut self) {
        self.commitment = Commitment::ProviderStarted;
    }

    pub(crate) fn transfer_stream(&mut self) {
        assert_eq!(self.operation, Operation::Send);
        self.commitment = Commitment::ProviderStarted;
        self.operation = Operation::StreamComplete;
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn failure_stage(&self) -> Option<FailureStage> {
        self.failure_stage
    }

    pub(crate) fn advance(
        &mut self,
        outcome: Outcome,
        observations: Observations,
    ) -> Option<Transition> {
        use Operation::*;

        if matches!(self.operation, Complete(_)) {
            return None;
        }
        self.generation += 1;
        let current = self.operation;
        let failure = if observations.logger_available
            && !(self.options.asynchronous && self.options.internal_call)
        {
            SyncFailure
        } else {
            Restore
        };
        let error = if outcome != Outcome::Success
            && current
                .contract(CallbackRuntime::Native, self.options.asynchronous)
                .failure
                != FailurePolicy::PreserveOriginalFailure
        {
            self.outcome = outcome;
            ErrorDisposition::Replace
        } else {
            ErrorDisposition::Preserve
        };
        let failure_stage =
            (outcome != Outcome::Success).then(|| self.classify_failure_stage(current));
        if error == ErrorDisposition::Replace {
            self.failure_stage = failure_stage;
        }
        if matches!(current, Send | StreamComplete) {
            self.commitment = if outcome == Outcome::Success {
                Commitment::ResponseReceived
            } else {
                Commitment::ProviderStarted
            };
        }
        self.operation = match (current, outcome) {
            (Restore, _) => Complete(self.outcome),
            (DeploymentFailure, _) => failure,
            (_, Outcome::Abort) => Restore,
            (SyncFailure | AsyncFailure, Outcome::Failure) => Restore,
            (operation, Outcome::Failure)
                if self.options.asynchronous && operation.notifies_deployment_failure() =>
            {
                DeploymentFailure
            }
            (_, Outcome::Failure) => failure,
            (Setup, Outcome::Success) if self.options.asynchronous => DeploymentPre,
            (Setup | DeploymentPre, Outcome::Success) => InputHooks,
            (InputHooks, Outcome::Success) => BuildRequest,
            (BuildRequest, Outcome::Success) => PreCall,
            (PreCall, Outcome::Success) => Send,
            (Send, Outcome::Success) if self.options.asynchronous => DeploymentSuccess,
            (Send, Outcome::Success) => SyncSuccess,
            (StreamComplete, Outcome::Success) if !self.options.asynchronous => SyncSuccess,
            (StreamComplete, Outcome::Success)
                if self.options.internal_call || observations.has_fallbacks =>
            {
                SyncSuccessIfNeeded
            }
            (StreamComplete, Outcome::Success) => AsyncSuccess,
            (DeploymentSuccess, Outcome::Success) => {
                if self.options.internal_call || observations.has_fallbacks {
                    SyncSuccessIfNeeded
                } else {
                    AsyncSuccess
                }
            }
            (AsyncSuccess, Outcome::Success) => SyncSuccessIfNeeded,
            (SyncFailure, Outcome::Success) if self.options.asynchronous => AsyncFailure,
            (SyncSuccess | SyncSuccessIfNeeded | SyncFailure | AsyncFailure, Outcome::Success) => {
                Restore
            }
            (Complete(_), _) => unreachable!(),
        };
        Some(Transition {
            operation: self.operation,
            error,
            commitment: self.commitment,
            failure_stage: self.failure_stage,
        })
    }

    pub(crate) fn delivers_terminal(&self, observations: Observations) -> bool {
        observations.logger_available
            && match self.operation {
                Operation::SyncSuccess | Operation::AsyncSuccess | Operation::SyncFailure => true,
                Operation::SyncSuccessIfNeeded => {
                    self.options.internal_call || observations.has_fallbacks
                }
                _ => false,
            }
    }

    fn classify_failure_stage(&self, operation: Operation) -> FailureStage {
        match (self.commitment, operation) {
            (Commitment::BeforeProvider, Operation::Send) => FailureStage::ProviderCall,
            (Commitment::BeforeProvider, _) => FailureStage::BeforeProvider,
            (Commitment::ProviderStarted, _) => FailureStage::ProviderCall,
            (Commitment::ResponseReceived, _) => FailureStage::AfterProviderResponse,
        }
    }
}

impl Default for CallLifecycle {
    fn default() -> Self {
        Self::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        })
    }
}

impl Operation {
    pub fn notifies_deployment_failure(self) -> bool {
        matches!(
            self,
            Self::InputHooks | Self::BuildRequest | Self::PreCall | Self::Send
        )
    }

    pub fn contract(self, runtime: CallbackRuntime, asynchronous: bool) -> OperationContract {
        use Operation::*;
        let delivery = match self {
            DeploymentPre | DeploymentSuccess | DeploymentFailure | AsyncFailure => {
                Delivery::InlineAwaited
            }
            Send if asynchronous => Delivery::InlineAwaited,
            InputHooks | PreCall if runtime == CallbackRuntime::Native => Delivery::InlineAwaited,
            SyncSuccess | SyncSuccessIfNeeded => Delivery::BlockingWorker,
            AsyncSuccess => Delivery::BackgroundTask,
            StreamComplete => Delivery::Deferred,
            _ => Delivery::InlineDirect,
        };
        let result = match self {
            DeploymentPre | DeploymentSuccess | InputHooks | BuildRequest | Send => {
                ResultPolicy::Transform
            }
            PreCall if runtime == CallbackRuntime::Native => ResultPolicy::Transform,
            _ => ResultPolicy::Observe,
        };
        let failure = match self {
            DeploymentFailure => FailurePolicy::PreserveOriginalFailure,
            SyncSuccess | AsyncSuccess | SyncSuccessIfNeeded
                if runtime == CallbackRuntime::Native =>
            {
                FailurePolicy::RecordAndContinue
            }
            _ => FailurePolicy::Propagate,
        };
        OperationContract {
            delivery,
            result,
            failure,
        }
    }

    pub fn is_awaited(self, asynchronous: bool) -> bool {
        self.contract(CallbackRuntime::Python, asynchronous)
            .is_awaited()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations() -> Observations {
        Observations {
            logger_available: true,
            has_fallbacks: false,
        }
    }

    #[test]
    fn commitment_classifies_failures_without_host_inference() {
        let mut before = CallLifecycle::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        });
        let failure = before.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::BeforeProvider);
        assert_eq!(failure.failure_stage, Some(FailureStage::BeforeProvider));

        let mut provider = CallLifecycle::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        });
        provider.advance(Outcome::Success, observations()).unwrap();
        provider.advance(Outcome::Success, observations()).unwrap();
        provider.advance(Outcome::Success, observations()).unwrap();
        provider.advance(Outcome::Success, observations()).unwrap();
        let failure = provider.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::ProviderStarted);
        assert_eq!(failure.failure_stage, Some(FailureStage::ProviderCall));

        let mut after = CallLifecycle::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        });
        after.advance(Outcome::Success, observations()).unwrap();
        after.advance(Outcome::Success, observations()).unwrap();
        after.advance(Outcome::Success, observations()).unwrap();
        after.advance(Outcome::Success, observations()).unwrap();
        after.advance(Outcome::Success, observations()).unwrap();
        let failure = after.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::ResponseReceived);
        assert_eq!(
            failure.failure_stage,
            Some(FailureStage::AfterProviderResponse)
        );
    }

    #[test]
    fn request_build_and_pre_call_failures_are_before_provider() {
        for success_count in [1, 2, 3] {
            let mut program = CallLifecycle::planned(ProgramOptions {
                asynchronous: false,
                internal_call: false,
            });
            for _ in 0..success_count {
                program.advance(Outcome::Success, observations()).unwrap();
            }
            let failure = program.advance(Outcome::Failure, observations()).unwrap();
            assert_eq!(failure.commitment, Commitment::BeforeProvider);
            assert_eq!(failure.failure_stage, Some(FailureStage::BeforeProvider));
        }
    }

    #[test]
    fn tickets_reject_reentry_foreign_owners_and_duplicate_completion() {
        let mut first = CallLifecycle::asynchronous();
        let mut second = CallLifecycle::asynchronous();
        let ticket = first.issue().unwrap();
        let second_ticket = second.issue().unwrap();
        assert!(first.issue().is_err());
        assert!(
            second
                .complete_operation(ticket, Outcome::Success, observations())
                .is_err()
        );
        assert_eq!(second.operation(), Operation::Setup);
        let ticket = second_ticket;
        let duplicate = OperationTicket {
            owner: ticket.owner,
            generation: ticket.generation,
            operation: ticket.operation,
        };
        second
            .complete_operation(ticket, Outcome::Success, observations())
            .unwrap();
        assert!(
            second
                .complete_operation(duplicate, Outcome::Success, observations())
                .is_err()
        );
        assert_eq!(second.operation(), Operation::DeploymentPre);
    }

    #[test]
    fn provider_permit_is_single_use_and_requires_send() {
        let mut program = CallLifecycle::asynchronous();
        let ticket = program.issue().unwrap();
        assert!(
            program
                .provider_permit::<crate::messages::lifecycle::MessagesRoute>(&ticket)
                .is_err()
        );
        program
            .complete_operation(ticket, Outcome::Success, observations())
            .unwrap();
        while program.operation() != Operation::Send {
            let ticket = program.issue().unwrap();
            program
                .complete_operation(ticket, Outcome::Success, observations())
                .unwrap();
        }
        let ticket = program.issue().unwrap();
        let _permit = program
            .provider_permit::<crate::messages::lifecycle::MessagesRoute>(&ticket)
            .unwrap();
        assert_eq!(program.commitment(), Commitment::ProviderStarted);
        assert!(
            program
                .provider_permit::<crate::messages::lifecycle::MessagesRoute>(&ticket)
                .is_err()
        );
    }
}
