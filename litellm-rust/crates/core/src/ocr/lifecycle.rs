use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::handler::perform_ocr_request;
use super::hooks::{
    OcrDuringCallRequest, OcrHookFuture, OcrHooks, OcrPostCallRequest, OcrPreCallRequest,
};
use super::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrClient};
use crate::AuthError;
use crate::Error;
use crate::auth::{ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};
use crate::call_lifecycle::execution::HostExchange;
use crate::call_lifecycle::host::{
    HostCall, HostCallFuture, HostCallStep, HostFailure, HostPhase, LifecycleBackend,
    LifecycleBackendFuture,
};
use crate::call_lifecycle::workflow::{
    LifecycleCall, LifecycleOperation, Workflow, WorkflowFuture, WorkflowReply,
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

pub struct OcrCall(LifecycleCall<OcrWorkflow>);

impl OcrCall {
    pub fn admit(client: OcrClient, admission: OcrAdmission) -> NativeOutcome<Self> {
        if !admission.provider_workflow {
            return NativeOutcome::Declined(OcrDecline::ProviderWorkflow);
        }
        if !admission.host_operations {
            return NativeOutcome::Declined(OcrDecline::HostOperations);
        }
        NativeOutcome::Completed(Self(LifecycleCall::new(
            OcrWorkflow {
                client,
                terminal: Arc::default(),
            },
            admission.asynchronous,
        )))
    }

    pub async fn resume(&mut self, result: Option<OcrHostResult>) -> Result<OcrCallStep, Error> {
        self.0.resume(result).await
    }

    pub async fn interrupt(&mut self, failure: HostFailure) -> Result<OcrCallStep, Error> {
        self.0.interrupt(failure).await
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

struct OcrWorkflow {
    client: OcrClient,
    terminal: Arc<std::sync::Mutex<Option<(CallLifecycleContext, CallLifecycleTiming)>>>,
}

impl Workflow for OcrWorkflow {
    type Request = (Box<LiteLLMOcrRequest>, bool);
    type Operation = OcrHostOperation;
    type Reply = OcrHostResult;
    type Response = LiteLLMOcrResponse;

    fn operation(operation: LifecycleOperation<Self::Response>) -> Self::Operation {
        match operation {
            LifecycleOperation::ProjectRequest => OcrHostOperation::ProjectRequest,
            LifecycleOperation::Phase(phase) => OcrHostOperation::Lifecycle(phase),
            LifecycleOperation::ConstructResponse(response) => {
                OcrHostOperation::ConstructResponse(response)
            }
            LifecycleOperation::ConstructCachedResponse(_) => {
                unreachable!("OCR does not cache responses")
            }
            LifecycleOperation::MapFailure(error) => OcrHostOperation::MapFailure(error),
            LifecycleOperation::Success {
                context,
                response,
                timing,
            } => OcrHostOperation::Success {
                context,
                response,
                timing,
            },
            LifecycleOperation::Failure {
                context,
                error,
                timing,
            } => OcrHostOperation::Failure {
                context,
                error,
                timing,
            },
        }
    }

    fn accepts(operation: &OcrHostOperation, reply: &OcrHostResult) -> bool {
        matches!(reply, OcrHostResult::Lifecycle(Err(_)))
            || matches!(
                (operation, reply),
                (OcrHostOperation::ProjectRequest, OcrHostResult::Request(_))
                    | (
                        OcrHostOperation::AcquireAzureAdToken,
                        OcrHostResult::AzureAdToken(_)
                    )
                    | (OcrHostOperation::PreCall(_), OcrHostResult::PreCall(_))
                    | (
                        OcrHostOperation::DuringCall(_),
                        OcrHostResult::DuringCall(_)
                    )
                    | (OcrHostOperation::PostCall(_), OcrHostResult::PostCall(_))
                    | (
                        OcrHostOperation::Lifecycle(_)
                            | OcrHostOperation::ConstructResponse(_)
                            | OcrHostOperation::MapFailure(_)
                            | OcrHostOperation::Success { .. }
                            | OcrHostOperation::Failure { .. },
                        OcrHostResult::Lifecycle(_)
                    )
            )
    }

    fn reply(&mut self, reply: OcrHostResult) -> WorkflowReply<Self::Request, OcrHostResult> {
        match reply {
            OcrHostResult::Request(request) => WorkflowReply::Request(request),
            OcrHostResult::Lifecycle(result) => WorkflowReply::Lifecycle(result),
            reply => WorkflowReply::Operation(reply),
        }
    }

    fn start(
        &mut self,
        (request, azure_ad_token_provider): Self::Request,
        operations: HostExchange<OcrHostOperation, OcrHostResult>,
    ) -> WorkflowFuture<LiteLLMOcrResponse> {
        let client = self.client.clone();
        let mut request = *request;
        let intercepts_requests = request.hooks.intercepts_requests();
        let context = CallLifecycleContext::new(
            "ocr",
            request.model.clone(),
            request.adapter.provider().as_str(),
            request
                .litellm_call_id
                .clone()
                .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
        );
        let started = epoch_seconds();
        if azure_ad_token_provider {
            request.azure_ad_token_provider = Some(TokenProviderHandle::new(Arc::new(
                OcrAzureAdTokenProvider {
                    operations: operations.clone(),
                },
            )));
        }
        request.hooks = Arc::new(ProtocolHooks {
            operations,
            intercepts_requests,
        });
        let terminal = self.terminal.clone();
        Box::pin(async move {
            let result = perform_ocr_request(&client, request).await;
            let timing = CallLifecycleTiming::new(started, epoch_seconds());
            *terminal.lock().unwrap_or_else(|error| error.into_inner()) = Some((context, timing));
            result
        })
    }

    fn terminal(&self) -> Option<(CallLifecycleContext, CallLifecycleTiming)> {
        self.terminal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

struct ProtocolHooks {
    operations: HostExchange<OcrHostOperation, OcrHostResult>,
    intercepts_requests: bool,
}

#[derive(Debug)]
struct OcrAzureAdTokenProvider {
    operations: HostExchange<OcrHostOperation, OcrHostResult>,
}

impl TokenProvider for OcrAzureAdTokenProvider {
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin(async move {
            match self
                .operations
                .invoke(OcrHostOperation::AcquireAzureAdToken)
                .await
                .map_err(|error| AuthError::AzureTokenAcquisition(error.to_string()))?
            {
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
        self.operations.invoke(operation).await
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
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

pub struct NoopOcrHost;

impl LifecycleBackend<OcrHostOperation, OcrHostResult> for NoopOcrHost {
    fn invoke(&self, operation: OcrHostOperation) -> LifecycleBackendFuture<'_, OcrHostResult> {
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

impl LifecycleBackend<OcrHostOperation, OcrHostResult> for OcrHookHost {
    fn invoke(&self, operation: OcrHostOperation) -> LifecycleBackendFuture<'_, OcrHostResult> {
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
