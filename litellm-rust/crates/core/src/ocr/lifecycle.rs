use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

use super::handler::perform_ocr_request;
use super::hooks::{
    OcrDuringCallRequest, OcrHookFuture, OcrHooks, OcrLogFuture, OcrPostCallRequest,
    OcrPreCallRequest,
};
use super::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrClient};
use crate::AuthError;
use crate::Error;
use crate::auth::{ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};
use crate::call_lifecycle::host::{
    HostCall, HostCallFuture, HostCallStep, HostFailure, HostLifecycle, HostPhase,
};
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};

pub type NativeResult<T> = Result<NativeOutcome<T>, Error>;

#[derive(Debug, PartialEq, Eq)]
pub enum NativeOutcome<T> {
    Completed(T),
    Declined(OcrDecline),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDecline {
    ProviderWorkflow,
    HostOperations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OcrAdmission {
    pub provider_workflow: bool,
    pub host_operations: bool,
    pub asynchronous: bool,
}

impl OcrAdmission {
    pub const fn all() -> Self {
        Self {
            provider_workflow: true,
            host_operations: true,
            asynchronous: false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum OcrHostOperation {
    ProjectRequest,
    Lifecycle(HostPhase),
    ConstructResponse(Arc<LiteLLMOcrResponse>),
    MapFailure(Error),
    Success {
        context: CallLifecycleContext,
        response: Arc<LiteLLMOcrResponse>,
        timing: CallLifecycleTiming,
    },
    Failure {
        context: CallLifecycleContext,
        error: Error,
        timing: CallLifecycleTiming,
    },
    AcquireAzureAdToken,
    PreCall(OcrPreCallRequest),
    DuringCall(OcrDuringCallRequest),
    PostCall(OcrPostCallRequest),
}

impl OcrHostOperation {
    pub const fn phase(&self) -> Option<HostPhase> {
        match self {
            Self::Lifecycle(phase) => Some(*phase),
            Self::Success { .. } => Some(HostPhase::Success),
            Self::Failure { .. } => Some(HostPhase::Failure),
            _ => None,
        }
    }
}

pub enum OcrHostResult {
    Request(Result<(Box<LiteLLMOcrRequest>, bool), Error>),
    Lifecycle(Result<(), HostFailure>),
    AzureAdToken(Result<ResolvedCredential, AuthError>),
    PreCall(Result<OcrPreCallRequest, Error>),
    DuringCall(Result<OcrDuringCallRequest, Error>),
    PostCall(Result<OcrPostCallRequest, Error>),
}

pub type OcrCallStep = HostCallStep<OcrHostOperation, LiteLLMOcrResponse>;

pub struct OcrCall {
    lifecycle: HostLifecycle,
    execution: OcrExecution,
    response: Option<Arc<LiteLLMOcrResponse>>,
    error: Option<Error>,
    pending: bool,
    completed: bool,
    projecting: bool,
}

impl OcrCall {
    pub fn admit(client: OcrClient, admission: OcrAdmission) -> NativeOutcome<Self> {
        if !admission.provider_workflow {
            return NativeOutcome::Declined(OcrDecline::ProviderWorkflow);
        }
        if !admission.host_operations {
            return NativeOutcome::Declined(OcrDecline::HostOperations);
        }
        NativeOutcome::Completed(Self {
            lifecycle: HostLifecycle::new(admission.asynchronous),
            execution: OcrExecution::new(client),
            response: None,
            error: None,
            pending: false,
            completed: false,
            projecting: false,
        })
    }

    pub async fn resume(&mut self, result: Option<OcrHostResult>) -> Result<OcrCallStep, Error> {
        if self.completed {
            return Err(Error::InvalidRequest(
                "OCR call cannot be resumed after completion".into(),
            ));
        }
        if self.pending != result.is_some() {
            return Err(Error::InvalidRequest(
                "OCR host operation result does not match pending state".into(),
            ));
        }
        match &result {
            Some(OcrHostResult::Lifecycle(Ok(())))
                if self.lifecycle.phase() == HostPhase::Execute =>
            {
                return Err(Error::InvalidRequest(
                    "OCR provider operation requires a typed result".into(),
                ));
            }
            Some(result)
                if !matches!(result, OcrHostResult::Lifecycle(_))
                    && self.lifecycle.phase() != HostPhase::Execute =>
            {
                return Err(Error::InvalidRequest(
                    "unexpected OCR provider operation result".into(),
                ));
            }
            _ => {}
        }
        self.pending = false;
        let provider_result = match result {
            Some(OcrHostResult::Request(result)) if self.projecting => {
                self.projecting = false;
                match result {
                    Ok((request, azure_ad_token_provider)) => {
                        self.execution.request = Some(*request);
                        self.execution.azure_ad_token_provider = azure_ad_token_provider;
                    }
                    Err(error) => self.accept(Err(HostFailure::Error(error))),
                }
                None
            }
            Some(OcrHostResult::Request(_)) => {
                return Err(Error::InvalidRequest(
                    "unexpected OCR request projection".into(),
                ));
            }
            Some(OcrHostResult::Lifecycle(result)) => {
                self.accept(result);
                None
            }
            result => result,
        };
        if self.lifecycle.phase() == HostPhase::Execute {
            if self.execution.request.is_none()
                && self.execution.execution.is_none()
                && !self.execution.completed
            {
                self.projecting = true;
                return Ok(self.host_step(OcrHostOperation::ProjectRequest));
            }
            match self.execution.resume(provider_result).await {
                Ok(OcrCallStep::Host(operation)) => return Ok(self.host_step(operation)),
                Ok(OcrCallStep::Complete(response)) => {
                    self.response = Some(Arc::new(response));
                    self.accept(Ok(()));
                }
                Err(error) => self.accept(Err(HostFailure::Error(error))),
            }
        }
        if self.error.is_some() {
            self.execution.stop().await;
        }
        let operation = match self.lifecycle.phase() {
            HostPhase::Complete => {
                self.completed = true;
                return match self.error.take() {
                    Some(error) => Err(error),
                    None => self
                        .response
                        .take()
                        .map(Arc::unwrap_or_clone)
                        .map(OcrCallStep::Complete)
                        .ok_or_else(|| {
                            Error::InvalidRequest("OCR completed without a response".into())
                        }),
                };
            }
            HostPhase::ConstructResponse => OcrHostOperation::ConstructResponse(
                self.response
                    .as_ref()
                    .ok_or_else(|| Error::InvalidRequest("missing OCR response".into()))?
                    .clone(),
            ),
            HostPhase::MapFailure => OcrHostOperation::MapFailure(
                self.error
                    .as_ref()
                    .ok_or_else(|| Error::InvalidRequest("missing OCR failure".into()))?
                    .clone(),
            ),
            HostPhase::Success | HostPhase::Failure => {
                let snapshot = self
                    .execution
                    .terminal
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone();
                match (self.lifecycle.phase(), snapshot) {
                    (HostPhase::Success, Some((context, timing))) => OcrHostOperation::Success {
                        context,
                        response: self
                            .response
                            .as_ref()
                            .ok_or_else(|| Error::InvalidRequest("missing OCR response".into()))?
                            .clone(),
                        timing,
                    },
                    (HostPhase::Failure, Some((context, timing))) => OcrHostOperation::Failure {
                        context,
                        error: self
                            .error
                            .as_ref()
                            .ok_or_else(|| Error::InvalidRequest("missing OCR failure".into()))?
                            .clone(),
                        timing,
                    },
                    (phase, _) => OcrHostOperation::Lifecycle(phase),
                }
            }
            phase => OcrHostOperation::Lifecycle(phase),
        };
        Ok(self.host_step(operation))
    }

    fn accept(&mut self, result: Result<(), HostFailure>) {
        let cancelled = matches!(&result, Err(HostFailure::Cancelled(_)));
        if let Some(error) = self.lifecycle.accept(result) {
            if cancelled {
                self.error = Some(error);
            } else {
                self.error.get_or_insert(error);
            }
            self.execution.cancel();
        }
    }

    pub async fn interrupt(&mut self, failure: HostFailure) -> Result<OcrCallStep, Error> {
        if self.completed {
            return Err(Error::InvalidRequest(
                "OCR call cannot be interrupted after completion".into(),
            ));
        }
        self.pending = false;
        self.accept(Err(failure));
        self.resume(None).await
    }

    fn host_step(&mut self, operation: OcrHostOperation) -> OcrCallStep {
        self.pending = true;
        OcrCallStep::Host(operation)
    }
}

impl HostCall for OcrCall {
    type Operation = OcrHostOperation;
    type Result = OcrHostResult;
    type Complete = LiteLLMOcrResponse;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete> {
        Box::pin(OcrCall::resume(self, result))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete> {
        Box::pin(OcrCall::interrupt(self, failure))
    }
}

struct PendingOperation {
    operation: OcrHostOperation,
    result: oneshot::Sender<OcrHostResult>,
}

struct OcrExecution {
    client: Option<OcrClient>,
    request: Option<LiteLLMOcrRequest>,
    operations_tx: mpsc::UnboundedSender<PendingOperation>,
    operations_rx: mpsc::UnboundedReceiver<PendingOperation>,
    pending_result: Option<oneshot::Sender<OcrHostResult>>,
    execution: Option<tokio::task::JoinHandle<Result<LiteLLMOcrResponse, Error>>>,
    completed: bool,
    azure_ad_token_provider: bool,
    terminal: Arc<std::sync::Mutex<Option<(CallLifecycleContext, CallLifecycleTiming)>>>,
}

impl OcrExecution {
    fn new(client: OcrClient) -> Self {
        let (operations_tx, operations_rx) = mpsc::unbounded_channel();
        Self {
            client: Some(client),
            request: None,
            operations_tx,
            operations_rx,
            pending_result: None,
            execution: None,
            completed: false,
            azure_ad_token_provider: false,
            terminal: Arc::default(),
        }
    }

    pub async fn resume(&mut self, result: Option<OcrHostResult>) -> Result<OcrCallStep, Error> {
        if self.completed {
            return Err(Error::InvalidRequest(
                "OCR call cannot be resumed after completion".into(),
            ));
        }
        match (self.pending_result.take(), result) {
            (Some(sender), Some(result)) => sender
                .send(result)
                .map_err(|_| Error::InvalidRequest("OCR host operation was abandoned".into()))?,
            (None, None) if self.execution.is_none() => self.start(),
            (Some(sender), None) => {
                self.pending_result = Some(sender);
                return Err(Error::InvalidRequest(
                    "OCR host operation result is required".into(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::InvalidRequest(
                    "unexpected OCR host operation result".into(),
                ));
            }
            (None, None) => {}
        }

        let execution = self.execution.as_mut().ok_or_else(|| {
            Error::InvalidRequest("OCR call cannot be resumed after completion".into())
        })?;
        tokio::select! {
            operation = self.operations_rx.recv() => {
                let operation = operation.ok_or_else(|| Error::InvalidRequest("OCR operation channel closed".into()))?;
                self.pending_result = Some(operation.result);
                Ok(OcrCallStep::Host(operation.operation))
            }
            result = execution => {
                self.execution = None;
                self.completed = true;
                result
                    .map_err(|error| Error::Network(format!("OCR execution task failed: {error}")))?
                    .map(OcrCallStep::Complete)
            }
        }
    }

    fn start(&mut self) {
        let client = self.client.take().expect("admitted OCR call has a client");
        let mut request = self
            .request
            .take()
            .expect("admitted OCR call has a request");
        let intercepts_requests = request.hooks.intercepts_requests();
        if self.azure_ad_token_provider {
            request.azure_ad_token_provider = Some(TokenProviderHandle::new(Arc::new(
                OcrAzureAdTokenProvider {
                    operations: self.operations_tx.clone(),
                },
            )));
        }
        request.hooks = Arc::new(ProtocolHooks {
            operations: self.operations_tx.clone(),
            intercepts_requests,
            terminal: self.terminal.clone(),
        });
        self.execution = Some(tokio::spawn(async move {
            perform_ocr_request(&client, request).await
        }));
    }

    fn cancel(&mut self) {
        self.pending_result = None;
        if let Some(execution) = &self.execution {
            execution.abort();
        }
    }

    async fn stop(&mut self) {
        self.cancel();
        if let Some(execution) = self.execution.as_mut() {
            let _ = execution.await;
        }
        self.execution = None;
    }
}

impl Drop for OcrExecution {
    fn drop(&mut self) {
        if let Some(execution) = &self.execution {
            execution.abort();
        }
    }
}

struct ProtocolHooks {
    operations: mpsc::UnboundedSender<PendingOperation>,
    intercepts_requests: bool,
    terminal: Arc<std::sync::Mutex<Option<(CallLifecycleContext, CallLifecycleTiming)>>>,
}

#[derive(Debug)]
struct OcrAzureAdTokenProvider {
    operations: mpsc::UnboundedSender<PendingOperation>,
}

impl TokenProvider for OcrAzureAdTokenProvider {
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin(async move {
            let (result, receiver) = oneshot::channel();
            self.operations
                .send(PendingOperation {
                    operation: OcrHostOperation::AcquireAzureAdToken,
                    result,
                })
                .map_err(|_| {
                    AuthError::AzureTokenAcquisition("OCR host driver was abandoned".into())
                })?;
            match receiver.await.map_err(|_| {
                AuthError::AzureTokenAcquisition(
                    "OCR token provider operation was abandoned".into(),
                )
            })? {
                OcrHostResult::AzureAdToken(result) => result,
                _ => Err(AuthError::AzureTokenAcquisition(
                    "invalid OCR token provider host result".into(),
                )),
            }
        })
    }
}

impl ProtocolHooks {
    async fn invoke(&self, operation: OcrHostOperation) -> Result<OcrHostResult, Error> {
        let (result, receiver) = oneshot::channel();
        self.operations
            .send(PendingOperation { operation, result })
            .map_err(|_| Error::InvalidRequest("OCR host driver was abandoned".into()))?;
        receiver
            .await
            .map_err(|_| Error::InvalidRequest("OCR host operation was abandoned".into()))
    }
}

impl OcrHooks for ProtocolHooks {
    fn intercepts_requests(&self) -> bool {
        self.intercepts_requests
    }

    fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
        Box::pin(async move {
            match self.invoke(OcrHostOperation::PreCall(request)).await? {
                OcrHostResult::PreCall(result) => result,
                _ => Err(Error::InvalidRequest(
                    "invalid OCR pre-call host result".into(),
                )),
            }
        })
    }

    fn during_call(
        &self,
        request: OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
        Box::pin(async move {
            match self.invoke(OcrHostOperation::DuringCall(request)).await? {
                OcrHostResult::DuringCall(result) => result,
                _ => Err(Error::InvalidRequest(
                    "invalid OCR during-call host result".into(),
                )),
            }
        })
    }

    fn post_call(&self, request: OcrPostCallRequest) -> OcrHookFuture<'_, OcrPostCallRequest> {
        Box::pin(async move {
            match self.invoke(OcrHostOperation::PostCall(request)).await? {
                OcrHostResult::PostCall(result) => result,
                _ => Err(Error::InvalidRequest(
                    "invalid OCR post-call host result".into(),
                )),
            }
        })
    }

    fn success<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        _response: &'a LiteLLMOcrResponse,
        timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            *self
                .terminal
                .lock()
                .unwrap_or_else(|error| error.into_inner()) =
                Some((context.clone(), timing.clone()));
        })
    }

    fn failure<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        _error: &'a Error,
        timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            *self
                .terminal
                .lock()
                .unwrap_or_else(|error| error.into_inner()) =
                Some((context.clone(), timing.clone()));
        })
    }
}

pub type OcrHostFuture<'a> = Pin<Box<dyn Future<Output = OcrHostResult> + Send + 'a>>;

pub trait OcrHost: Send + Sync {
    fn invoke(&self, operation: OcrHostOperation) -> OcrHostFuture<'_>;
}

pub struct NoopOcrHost;

impl OcrHost for NoopOcrHost {
    fn invoke(&self, operation: OcrHostOperation) -> OcrHostFuture<'_> {
        Box::pin(async move {
            match operation {
                OcrHostOperation::ProjectRequest => OcrHostResult::Request(Err(
                    Error::InvalidRequest("OCR host has no request projection".into()),
                )),
                OcrHostOperation::Lifecycle(_)
                | OcrHostOperation::ConstructResponse(_)
                | OcrHostOperation::MapFailure(_)
                | OcrHostOperation::Success { .. }
                | OcrHostOperation::Failure { .. } => OcrHostResult::Lifecycle(Ok(())),
                OcrHostOperation::AcquireAzureAdToken => {
                    OcrHostResult::AzureAdToken(Err(AuthError::AzureTokenAcquisition(
                        "OCR host has no Azure AD token provider".into(),
                    )))
                }
                OcrHostOperation::PreCall(request) => OcrHostResult::PreCall(Ok(request)),
                OcrHostOperation::DuringCall(request) => OcrHostResult::DuringCall(Ok(request)),
                OcrHostOperation::PostCall(request) => OcrHostResult::PostCall(Ok(request)),
            }
        })
    }
}

pub struct OcrHookHost {
    hooks: Arc<dyn OcrHooks>,
}

impl OcrHookHost {
    pub fn new(hooks: Arc<dyn OcrHooks>) -> Self {
        Self { hooks }
    }
}

impl OcrHost for OcrHookHost {
    fn invoke(&self, operation: OcrHostOperation) -> OcrHostFuture<'_> {
        Box::pin(async move {
            match operation {
                OcrHostOperation::ProjectRequest => OcrHostResult::Request(Err(
                    Error::InvalidRequest("OCR hook host has no request projection".into()),
                )),
                OcrHostOperation::Success {
                    context,
                    response,
                    timing,
                } => {
                    self.hooks.success(&context, &response, &timing).await;
                    OcrHostResult::Lifecycle(Ok(()))
                }
                OcrHostOperation::Failure {
                    context,
                    error,
                    timing,
                } => {
                    self.hooks.failure(&context, &error, &timing).await;
                    OcrHostResult::Lifecycle(Ok(()))
                }
                OcrHostOperation::Lifecycle(_)
                | OcrHostOperation::ConstructResponse(_)
                | OcrHostOperation::MapFailure(_) => OcrHostResult::Lifecycle(Ok(())),
                OcrHostOperation::AcquireAzureAdToken => {
                    OcrHostResult::AzureAdToken(Err(AuthError::AzureTokenAcquisition(
                        "OCR hook host has no Azure AD token provider".into(),
                    )))
                }
                OcrHostOperation::PreCall(request) => {
                    OcrHostResult::PreCall(self.hooks.pre_call(request).await)
                }
                OcrHostOperation::DuringCall(request) => {
                    OcrHostResult::DuringCall(self.hooks.during_call(request).await)
                }
                OcrHostOperation::PostCall(request) => {
                    OcrHostResult::PostCall(self.hooks.post_call(request).await)
                }
            }
        })
    }
}
