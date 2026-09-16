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
use crate::call_lifecycle::host::{
    HostCall, HostCallFuture, HostCallStep, HostFailure, HostLifecycle, HostPhase,
};
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
use litellm_auth::{ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};

pub type NativeResult<T> = Result<NativeOutcome<T>, super::Error>;

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
    MapFailure(super::Error),
    Success {
        context: CallLifecycleContext,
        response: Arc<LiteLLMOcrResponse>,
        timing: CallLifecycleTiming,
    },
    Failure {
        context: CallLifecycleContext,
        error: super::Error,
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
    Request(Result<(Box<LiteLLMOcrRequest>, bool), super::Error>),
    Lifecycle(Result<(), HostFailure<super::Error>>),
    AzureAdToken(Result<ResolvedCredential, litellm_auth::Error>),
    PreCall(Result<OcrPreCallRequest, super::Error>),
    DuringCall(Result<OcrDuringCallRequest, super::Error>),
    PostCall(Result<OcrPostCallRequest, super::Error>),
}

pub type OcrCallStep = HostCallStep<OcrHostOperation, LiteLLMOcrResponse>;

pub struct OcrCall {
    lifecycle: HostLifecycle,
    execution: OcrExecution,
    response: Option<Arc<LiteLLMOcrResponse>>,
    error: Option<super::Error>,
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

    pub async fn resume(
        &mut self,
        result: Option<OcrHostResult>,
    ) -> Result<OcrCallStep, super::Error> {
        if self.completed {
            return Err(super::Error::InvalidRequest(
                "OCR call cannot be resumed after completion".into(),
            ));
        }
        if self.pending != result.is_some() {
            return Err(super::Error::InvalidRequest(
                "OCR host operation result does not match pending state".into(),
            ));
        }
        match &result {
            Some(OcrHostResult::Lifecycle(Ok(())))
                if self.lifecycle.phase() == HostPhase::Execute =>
            {
                return Err(super::Error::InvalidRequest(
                    "OCR provider operation requires a typed result".into(),
                ));
            }
            Some(result)
                if !matches!(result, OcrHostResult::Lifecycle(_))
                    && self.lifecycle.phase() != HostPhase::Execute =>
            {
                return Err(super::Error::InvalidRequest(
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
                return Err(super::Error::InvalidRequest(
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
                            super::Error::InvalidRequest("OCR completed without a response".into())
                        }),
                };
            }
            HostPhase::ConstructResponse => OcrHostOperation::ConstructResponse(
                self.response
                    .as_ref()
                    .ok_or_else(|| super::Error::InvalidRequest("missing OCR response".into()))?
                    .clone(),
            ),
            HostPhase::MapFailure => OcrHostOperation::MapFailure(
                self.error
                    .as_ref()
                    .ok_or_else(|| super::Error::InvalidRequest("missing OCR failure".into()))?
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
                            .ok_or_else(|| {
                                super::Error::InvalidRequest("missing OCR response".into())
                            })?
                            .clone(),
                        timing,
                    },
                    (HostPhase::Failure, Some((context, timing))) => OcrHostOperation::Failure {
                        context,
                        error: self
                            .error
                            .as_ref()
                            .ok_or_else(|| {
                                super::Error::InvalidRequest("missing OCR failure".into())
                            })?
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

    fn accept(&mut self, result: Result<(), HostFailure<super::Error>>) {
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

    pub async fn interrupt(
        &mut self,
        failure: HostFailure<super::Error>,
    ) -> Result<OcrCallStep, super::Error> {
        if self.completed {
            return Err(super::Error::InvalidRequest(
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
    type Error = super::Error;
    type Operation = OcrHostOperation;
    type Result = OcrHostResult;
    type Complete = LiteLLMOcrResponse;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete, Self::Error> {
        Box::pin(OcrCall::resume(self, result))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure<super::Error>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete, Self::Error> {
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
    execution: Option<tokio::task::JoinHandle<Result<LiteLLMOcrResponse, super::Error>>>,
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

    pub async fn resume(
        &mut self,
        result: Option<OcrHostResult>,
    ) -> Result<OcrCallStep, super::Error> {
        if self.completed {
            return Err(super::Error::InvalidRequest(
                "OCR call cannot be resumed after completion".into(),
            ));
        }
        match (self.pending_result.take(), result) {
            (Some(sender), Some(result)) => sender.send(result).map_err(|_| {
                super::Error::InvalidRequest("OCR host operation was abandoned".into())
            })?,
            (None, None) if self.execution.is_none() => self.start(),
            (Some(sender), None) => {
                self.pending_result = Some(sender);
                return Err(super::Error::InvalidRequest(
                    "OCR host operation result is required".into(),
                ));
            }
            (None, Some(_)) => {
                return Err(super::Error::InvalidRequest(
                    "unexpected OCR host operation result".into(),
                ));
            }
            (None, None) => {}
        }

        let execution = self.execution.as_mut().ok_or_else(|| {
            super::Error::InvalidRequest("OCR call cannot be resumed after completion".into())
        })?;
        tokio::select! {
            operation = self.operations_rx.recv() => {
                let operation = operation.ok_or_else(|| super::Error::InvalidRequest("OCR operation channel closed".into()))?;
                self.pending_result = Some(operation.result);
                Ok(OcrCallStep::Host(operation.operation))
            }
            result = execution => {
                self.execution = None;
                self.completed = true;
                result
                    .map_err(|error| super::Error::Transport(crate::transport::Error::Network(format!("OCR execution task failed: {error}"))))?
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
                    litellm_auth::Error::AzureTokenAcquisition(
                        "OCR host driver was abandoned".into(),
                    )
                })?;
            match receiver.await.map_err(|_| {
                litellm_auth::Error::AzureTokenAcquisition(
                    "OCR token provider operation was abandoned".into(),
                )
            })? {
                OcrHostResult::AzureAdToken(result) => result,
                _ => Err(litellm_auth::Error::AzureTokenAcquisition(
                    "invalid OCR token provider host result".into(),
                )),
            }
        })
    }
}

impl ProtocolHooks {
    async fn invoke(&self, operation: OcrHostOperation) -> Result<OcrHostResult, super::Error> {
        let (result, receiver) = oneshot::channel();
        self.operations
            .send(PendingOperation { operation, result })
            .map_err(|_| super::Error::InvalidRequest("OCR host driver was abandoned".into()))?;
        receiver
            .await
            .map_err(|_| super::Error::InvalidRequest("OCR host operation was abandoned".into()))
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
                _ => Err(super::Error::InvalidRequest(
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
                _ => Err(super::Error::InvalidRequest(
                    "invalid OCR during-call host result".into(),
                )),
            }
        })
    }

    fn post_call(&self, request: OcrPostCallRequest) -> OcrHookFuture<'_, OcrPostCallRequest> {
        Box::pin(async move {
            match self.invoke(OcrHostOperation::PostCall(request)).await? {
                OcrHostResult::PostCall(result) => result,
                _ => Err(super::Error::InvalidRequest(
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
        _error: &'a super::Error,
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
                    super::Error::InvalidRequest("OCR host has no request projection".into()),
                )),
                OcrHostOperation::Lifecycle(_)
                | OcrHostOperation::ConstructResponse(_)
                | OcrHostOperation::MapFailure(_)
                | OcrHostOperation::Success { .. }
                | OcrHostOperation::Failure { .. } => OcrHostResult::Lifecycle(Ok(())),
                OcrHostOperation::AcquireAzureAdToken => {
                    OcrHostResult::AzureAdToken(Err(litellm_auth::Error::AzureTokenAcquisition(
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
                    super::Error::InvalidRequest("OCR hook host has no request projection".into()),
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
                    OcrHostResult::AzureAdToken(Err(litellm_auth::Error::AzureTokenAcquisition(
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::{Value, json};

    use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
    use crate::ocr::OcrClient;
    use crate::ocr::hooks::{
        OcrDuringCallRequest, OcrHookFuture, OcrHooks, OcrLogFuture, OcrPostCallRequest,
        OcrPreCallRequest,
    };
    use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
    use crate::ocr::wire::{OcrWireRequest, decode_request};
    use crate::ocr::{
        NativeOutcome, NoopOcrHost, OcrAdmission, OcrCall, OcrCallStep, OcrDecline, OcrHost,
        OcrHostOperation, OcrHostResult,
    };

    #[test]
    fn request_boundary_selects_mistral_and_rejects_unknown_providers() {
        let request = OcrWireRequest {
            model: "mistral/model".into(),
            document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
            api_key: Some("key".into()),
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: json!({"extract_header":true,"unknown":42})
                .as_object()
                .unwrap()
                .clone()
                .into(),
            input_sources: Default::default(),
            timeout_seconds: None,
        };
        assert!(decode_request(request).is_ok());
        assert!(
            decode_request(OcrWireRequest {
                model: "model".into(),
                document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
                api_key: Some("key".into()),
                api_base: None,
                custom_llm_provider: Some("unknown".into()),
                extra_headers: None,
                optional_params: Default::default(),
                input_sources: Default::default(),
                timeout_seconds: None,
            })
            .is_err()
        );
    }

    #[tokio::test]
    async fn facade_executes_direct_mistral_once() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"hello","custom":"preserved"}],
            "usage_info":{"pages_processed":1}
        }))])
        .await;
        let result = perform_ocr(wire_request(
            "mistral/model",
            &base,
            json!({"pages":"0,2-4","extract_header":true,"unknown":{"nested":[null,false,0]}}),
        ))
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(result.pages[0].markdown, "hello");
        assert_eq!(result.pages[0].extra_fields["custom"], "preserved");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /v1/ocr "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key\r\n")
        );
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"model",
                "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "pages":"0,2-4",
                "extract_header":true,
                "unknown":{"nested":[null,false,0]}
            })
        );
    }

    #[tokio::test]
    async fn facade_resolves_dynamic_connection_before_auth_and_url_preparation() {
        use litellm_auth::{InputSource, Sourced};
        let (base, seen, server) =
            mock_server(vec![MockResponse::json(json!({"pages": []}))]).await;
        let request = wire_request("mistral/model", "https://unused.invalid", json!({}));
        let request = crate::ocr::LiteLLMOcrRequest {
            connection: crate::ocr::OcrConnection {
                dynamic_api_key: Some(Sourced::new("dynamic-key".into(), InputSource::Deployment)),
                dynamic_api_base: Some(Sourced::new(base, InputSource::Deployment)),
                ..request.connection
            },
            ..request
        };
        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /v1/ocr "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer dynamic-key\r\n")
        );
    }

    #[tokio::test]
    async fn provider_error_factory_preserves_status_body_and_response_headers() {
        let (base, seen, server) = mock_server(vec![MockResponse {
            status: 429,
            headers: vec![
                ("retry-after", "17".into()),
                ("x-request-id", "ocr-request".into()),
            ],
            body: json!({"message": "rate limited"}),
        }])
        .await;
        let error = perform_ocr(wire_request("mistral/model", &base, json!({})))
            .await
            .unwrap_err();
        server.await.unwrap();
        let crate::ocr::Error::Provider {
            status,
            body,
            headers,
        } = error
        else {
            panic!("expected provider error")
        };
        assert_eq!(status, 429);
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap(),
            json!({"message": "rate limited"})
        );
        assert!(headers.contains(&("retry-after".into(), "17".into())));
        assert!(headers.contains(&("x-request-id".into(), "ocr-request".into())));
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn facade_retains_native_response_when_requested() {
        let provider_response = json!({
            "pages":[{"index":0,"markdown":"hello"}],
            "usage_info":{"pages_processed":1},
            "provider_only":"preserved"
        });
        let (base, _, server) =
            mock_server(vec![MockResponse::json(provider_response.clone())]).await;
        let response = perform_ocr(wire_request(
            "mistral/model",
            &base,
            json!({"req_format":"native"}),
        ))
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(
            response.provider_native_response.as_ref(),
            provider_response.as_object()
        );
    }

    #[tokio::test]
    async fn facade_uses_the_injected_http_client() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            "x-transport-owner",
            reqwest::header::HeaderValue::from_static("host"),
        );
        let provider_http = reqwest::Client::builder()
            .default_headers(default_headers)
            .build()
            .unwrap();
        OcrClient::new(provider_http)
            .unwrap()
            .perform(wire_request("mistral/model", &base, json!({})))
            .await
            .unwrap();
        server.await.unwrap();
        assert!(seen.lock().unwrap()[0].contains("x-transport-owner: host"));
    }

    struct RecordingHooks {
        events: Arc<Mutex<Vec<&'static str>>>,
        block: bool,
    }

    struct ExtensionHooks;

    impl OcrHooks for ExtensionHooks {
        fn intercepts_requests(&self) -> bool {
            true
        }

        fn during_call(
            &self,
            mut request: OcrDuringCallRequest,
        ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
            Box::pin(async move {
                assert_eq!(request.body["pages"], json!([2]));
                assert_eq!(request.body.get("future"), Some(&Value::Null));
                assert!(
                    !request
                        .retained_fields
                        .iter()
                        .any(|field| field == "pages" || field == "document")
                );
                request.body.as_object_mut().unwrap().remove("future");
                request.body["hook_option"] = json!({"nested":[null,false,0]});
                Ok(request)
            })
        }
    }

    #[tokio::test]
    async fn composed_extensions_reach_hooks_and_removed_fields_stay_removed() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(ExtensionHooks),
            ..wire_request(
                "mistral/model",
                &base,
                json!({
                    "pages":[0], "future":null, "extra_body":{"pages":[2],
                        "document":{"type":"document_url","document_url":"data:application/pdf;base64,eHl6"}}
                }),
            )
        };
        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["pages"], json!([2]));
        assert_eq!(
            body["document"]["document_url"],
            "data:application/pdf;base64,eHl6"
        );
        assert_eq!(body["hook_option"], json!({"nested":[null,false,0]}));
        assert!(body.get("future").is_none());
    }

    impl OcrHooks for RecordingHooks {
        fn intercepts_requests(&self) -> bool {
            true
        }

        fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
            Box::pin(async move {
                self.events.lock().unwrap().push("pre");
                if self.block {
                    return Err(crate::ocr::Error::InvalidRequest("blocked".into()));
                }
                Ok(request)
            })
        }

        fn during_call(
            &self,
            request: crate::ocr::hooks::OcrDuringCallRequest,
        ) -> OcrHookFuture<'_, crate::ocr::hooks::OcrDuringCallRequest> {
            Box::pin(async move {
                self.events.lock().unwrap().push("during");
                Ok(request)
            })
        }

        fn post_call(&self, request: OcrPostCallRequest) -> OcrHookFuture<'_, OcrPostCallRequest> {
            Box::pin(async move {
                self.events.lock().unwrap().push("post");
                Ok(request)
            })
        }

        fn success<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            _response: &'a crate::ocr::LiteLLMOcrResponse,
            _timing: &'a CallLifecycleTiming,
        ) -> OcrLogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("success");
            })
        }

        fn failure<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            _error: &'a crate::ocr::Error,
            _timing: &'a CallLifecycleTiming,
        ) -> OcrLogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("failure");
            })
        }
    }

    struct HeaderEditHooks;

    impl OcrHooks for HeaderEditHooks {
        fn intercepts_requests(&self) -> bool {
            true
        }

        fn during_call(
            &self,
            mut request: OcrDuringCallRequest,
        ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
            request
                .headers
                .push(("x-core-callback".into(), "edited".into()));
            Box::pin(async move { Ok(request) })
        }
    }

    #[tokio::test]
    async fn lifecycle_sends_headers_returned_by_the_typed_during_call_operation() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(HeaderEditHooks),
            ..wire_request("mistral/model", &base, json!({}))
        };

        perform_ocr(request).await.unwrap();
        server.await.unwrap();

        assert!(seen.lock().unwrap()[0].contains("x-core-callback: edited"));
    }

    #[tokio::test]
    async fn lifecycle_orders_hooks_and_emits_one_success() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let request = wire_request("mistral/model", &base, json!({}));
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(RecordingHooks {
                events: events.clone(),
                block: false,
            }),
            ..request
        };
        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            *events.lock().unwrap(),
            ["pre", "during", "post", "success"]
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn lifecycle_blocking_prevents_execution_and_emits_one_failure() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let request = wire_request("mistral/model", "http://127.0.0.1:1", json!({}));
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(RecordingHooks {
                events: events.clone(),
                block: true,
            }),
            ..request
        };
        let error = perform_ocr(request).await.unwrap_err();
        assert!(matches!(error, crate::ocr::Error::InvalidRequest(_)));
        assert_eq!(*events.lock().unwrap(), ["pre", "failure"]);
    }

    #[tokio::test]
    async fn upstream_failure_emits_one_terminal_failure() {
        let (base, seen, server) = mock_server(vec![MockResponse {
            status: 500,
            headers: vec![],
            body: json!({"error":"failed"}),
        }])
        .await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let request = wire_request("mistral/model", &base, json!({}));
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(RecordingHooks {
                events: events.clone(),
                block: false,
            }),
            ..request
        };
        assert!(perform_ocr(request).await.is_err());
        server.await.unwrap();
        assert_eq!(*events.lock().unwrap(), ["pre", "during", "failure"]);
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    struct AdmissionSpy {
        effects: Arc<Mutex<usize>>,
    }

    impl OcrHooks for AdmissionSpy {
        fn intercepts_requests(&self) -> bool {
            *self.effects.lock().unwrap() += 1;
            true
        }

        fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
            *self.effects.lock().unwrap() += 1;
            Box::pin(async move { Ok(request) })
        }
    }

    #[test]
    fn admission_declines_without_invoking_hooks_or_transport() {
        for (admission, expected) in [
            (
                OcrAdmission {
                    provider_workflow: false,
                    host_operations: true,
                    asynchronous: false,
                },
                OcrDecline::ProviderWorkflow,
            ),
            (
                OcrAdmission {
                    provider_workflow: true,
                    host_operations: false,
                    asynchronous: false,
                },
                OcrDecline::HostOperations,
            ),
        ] {
            let outcome = OcrCall::admit(crate::ocr::test_support::ocr_client(), admission);
            assert!(matches!(outcome, NativeOutcome::Declined(reason) if reason == expected));
        }
    }

    #[tokio::test]
    async fn fallible_host_phases_do_not_replay_or_reach_transport() {
        for failure_phase in ["pre", "during"] {
            let request = crate::ocr::LiteLLMOcrRequest {
                hooks: Arc::new(AdmissionSpy {
                    effects: Arc::new(Mutex::new(0)),
                }),
                ..wire_request("mistral/model", "http://127.0.0.1:1", json!({}))
            };
            let NativeOutcome::Completed(mut call) =
                OcrCall::admit(crate::ocr::test_support::ocr_client(), OcrAdmission::all())
            else {
                panic!("supported call declined")
            };
            let mut request = Some(request);
            let mut result = None;
            let mut phases = Vec::new();
            let error = loop {
                match call.resume(result.take()).await {
                    Ok(OcrCallStep::Host(operation)) => match operation {
                        OcrHostOperation::Lifecycle(_)
                        | OcrHostOperation::ConstructResponse(_)
                        | OcrHostOperation::MapFailure(_)
                        | OcrHostOperation::Success { .. }
                        | OcrHostOperation::Failure { .. } => {
                            result = Some(OcrHostResult::Lifecycle(Ok(())))
                        }
                        OcrHostOperation::ProjectRequest => {
                            result = Some(OcrHostResult::Request(Ok((
                                Box::new(request.take().unwrap()),
                                false,
                            ))))
                        }
                        OcrHostOperation::AcquireAzureAdToken => {
                            panic!("test request has no token provider")
                        }
                        OcrHostOperation::PreCall(request) => {
                            phases.push("pre");
                            result = Some(OcrHostResult::PreCall(if failure_phase == "pre" {
                                Err(crate::ocr::Error::InvalidRequest("pre failed".into()))
                            } else {
                                Ok(request)
                            }));
                        }
                        OcrHostOperation::DuringCall(request) => {
                            phases.push("during");
                            result =
                                Some(OcrHostResult::DuringCall(if failure_phase == "during" {
                                    Err(crate::ocr::Error::InvalidRequest("during failed".into()))
                                } else {
                                    Ok(request)
                                }));
                        }
                        OcrHostOperation::PostCall(_) => panic!("transport should not be reached"),
                    },
                    Err(error) => break error,
                    Ok(OcrCallStep::Complete(_)) => panic!("failed call completed"),
                }
            };
            assert!(matches!(error, crate::ocr::Error::InvalidRequest(_)));
            assert_eq!(
                phases
                    .iter()
                    .filter(|phase| **phase == failure_phase)
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn invalid_provider_response_runs_post_call_before_normalization_failure() {
        let (base, seen, server) =
            mock_server(vec![MockResponse::json(json!({"pages":"invalid"}))]).await;
        let mut request = Some(wire_request("mistral/model", &base, json!({})));
        let NativeOutcome::Completed(mut call) =
            OcrCall::admit(crate::ocr::test_support::ocr_client(), OcrAdmission::all())
        else {
            panic!("supported call declined")
        };
        let host = NoopOcrHost;
        let mut result = None;
        let mut post_calls = Vec::new();
        let error = loop {
            match call.resume(result.take()).await {
                Ok(OcrCallStep::Host(OcrHostOperation::ProjectRequest)) => {
                    result = Some(OcrHostResult::Request(Ok((
                        Box::new(request.take().unwrap()),
                        false,
                    ))));
                }
                Ok(OcrCallStep::Host(operation)) => {
                    if let OcrHostOperation::PostCall(request) = &operation {
                        post_calls.push(request.original_response.clone());
                    }
                    result = Some(host.invoke(operation).await);
                }
                Err(error) => break error,
                Ok(OcrCallStep::Complete(_)) => panic!("invalid provider response completed"),
            }
        };
        server.await.unwrap();
        assert!(matches!(
            error,
            crate::ocr::Error::ResponseField { ref path } if path == "pages"
        ));
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert_eq!(post_calls, [json!(r#"{"pages":"invalid"}"#)]);
    }

    #[tokio::test]
    async fn direct_native_host_drives_the_same_state_machine() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"native"}]
        }))])
        .await;
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(AdmissionSpy {
                effects: Arc::new(Mutex::new(0)),
            }),
            ..wire_request("mistral/model", &base, json!({}))
        };
        let NativeOutcome::Completed(mut call) = OcrCall::admit(
            crate::ocr::test_support::ocr_client(),
            OcrAdmission {
                asynchronous: true,
                ..OcrAdmission::all()
            },
        ) else {
            panic!("supported call declined")
        };
        let mut request = Some(request);
        let host = NoopOcrHost;
        let mut result = None;
        let mut operations = Vec::new();
        let response = loop {
            match call.resume(result.take()).await.unwrap() {
                OcrCallStep::Host(operation) => {
                    operations.push(match &operation {
                        OcrHostOperation::ProjectRequest => "ProjectRequest".into(),
                        OcrHostOperation::Lifecycle(phase) => format!("{phase:?}"),
                        OcrHostOperation::PreCall(_) => "PreCall".into(),
                        OcrHostOperation::DuringCall(_) => "DuringCall".into(),
                        OcrHostOperation::PostCall(_) => "PostCall".into(),
                        OcrHostOperation::ConstructResponse(_) => "ConstructResponse".into(),
                        OcrHostOperation::Success { response, .. } => {
                            assert_eq!(response.pages[0].markdown, "native");
                            "Success".into()
                        }
                        _ => panic!("unexpected OCR operation"),
                    });
                    result = Some(match operation {
                        OcrHostOperation::ProjectRequest => {
                            OcrHostResult::Request(Ok((Box::new(request.take().unwrap()), false)))
                        }
                        operation => host.invoke(operation).await,
                    });
                }
                OcrCallStep::Complete(response) => break response,
            }
        };
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "native");
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert_eq!(
            operations,
            [
                "Setup",
                "DeploymentPreCall",
                "Prepare",
                "ProjectRequest",
                "PreCall",
                "DuringCall",
                "PostCall",
                "ConstructResponse",
                "DeploymentPostCall",
                "Finalize",
                "Success",
            ]
        );
        assert!(matches!(
            call.resume(None).await,
            Err(crate::ocr::Error::InvalidRequest(_))
        ));
    }

    #[tokio::test]
    async fn public_finalization_failure_never_dispatches_success_or_replays_provider() {
        use crate::call_lifecycle::host::{HostFailure, HostPhase};

        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut request = Some(wire_request("mistral/model", &base, json!({})));
        let NativeOutcome::Completed(mut call) = OcrCall::admit(
            crate::ocr::test_support::ocr_client(),
            OcrAdmission {
                asynchronous: true,
                ..OcrAdmission::all()
            },
        ) else {
            panic!("supported call declined")
        };
        let selected = crate::ocr::Error::InvalidRequest("public metadata failed".into());
        let host = NoopOcrHost;
        let mut result = None;
        let mut failures = Vec::new();
        let error = loop {
            match call.resume(result.take()).await {
                Ok(OcrCallStep::Host(operation)) => {
                    result = Some(match operation {
                        OcrHostOperation::Lifecycle(HostPhase::Finalize) => {
                            OcrHostResult::Lifecycle(Err(HostFailure::Error(selected.clone())))
                        }
                        OcrHostOperation::Failure { error, .. } => {
                            assert_eq!(error, selected);
                            failures.push("sync");
                            OcrHostResult::Lifecycle(Err(HostFailure::Error(
                                crate::ocr::Error::InvalidRequest("failure callback failed".into()),
                            )))
                        }
                        OcrHostOperation::Lifecycle(HostPhase::AsyncFailure) => {
                            failures.push("async");
                            OcrHostResult::Lifecycle(Ok(()))
                        }
                        OcrHostOperation::Success { .. }
                        | OcrHostOperation::MapFailure(_)
                        | OcrHostOperation::Lifecycle(HostPhase::DeploymentFailure) => {
                            panic!("finalization failure used provider/success dispatch")
                        }
                        OcrHostOperation::ProjectRequest => {
                            OcrHostResult::Request(Ok((Box::new(request.take().unwrap()), false)))
                        }
                        operation => host.invoke(operation).await,
                    });
                }
                Ok(OcrCallStep::Complete(_)) => panic!("failed call completed successfully"),
                Err(error) => break error,
            }
        };
        server.await.unwrap();
        assert_eq!(error, selected);
        assert_eq!(failures, ["sync", "async"]);
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn cancellation_at_provider_hook_prevents_execution_and_further_resumption() {
        use crate::call_lifecycle::host::HostFailure;

        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(AdmissionSpy {
                effects: Arc::new(Mutex::new(0)),
            }),
            ..wire_request("mistral/model", "http://127.0.0.1:1", json!({}))
        };
        let NativeOutcome::Completed(mut call) =
            OcrCall::admit(crate::ocr::test_support::ocr_client(), OcrAdmission::all())
        else {
            panic!("supported call declined")
        };
        let mut request = Some(request);
        let host = NoopOcrHost;
        let mut result = None;
        loop {
            match call.resume(result.take()).await.unwrap() {
                OcrCallStep::Host(OcrHostOperation::PreCall(_)) => break,
                OcrCallStep::Host(OcrHostOperation::ProjectRequest) => {
                    result = Some(OcrHostResult::Request(Ok((
                        Box::new(request.take().unwrap()),
                        false,
                    ))))
                }
                OcrCallStep::Host(operation) => result = Some(host.invoke(operation).await),
                OcrCallStep::Complete(_) => panic!("provider executed before pre-call result"),
            }
        }
        let selected = crate::ocr::Error::InvalidRequest("cancelled".into());
        assert!(matches!(
            call.interrupt(HostFailure::Cancelled(selected.clone())).await,
            Err(error) if error == selected
        ));
        assert!(
            call.resume(Some(OcrHostResult::Lifecycle(Ok(()))))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn missing_host_result_preserves_pending_operation() {
        use crate::call_lifecycle::host::HostPhase;

        let NativeOutcome::Completed(mut call) =
            OcrCall::admit(crate::ocr::test_support::ocr_client(), OcrAdmission::all())
        else {
            panic!("supported call declined")
        };
        assert!(matches!(
            call.resume(None).await.unwrap(),
            OcrCallStep::Host(OcrHostOperation::Lifecycle(HostPhase::Setup))
        ));
        assert!(call.resume(None).await.is_err());
        assert!(matches!(
            call.resume(Some(OcrHostResult::Lifecycle(Ok(()))))
                .await
                .unwrap(),
            OcrCallStep::Host(OcrHostOperation::Lifecycle(HostPhase::Prepare))
        ));
    }

    async fn read_bounded_response(
        response: Vec<u8>,
        limit: usize,
    ) -> Result<bytes::Bytes, crate::ocr::Error> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket.write_all(&response).await.unwrap();
            std::future::pending::<()>().await;
        });
        let response = reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            crate::ocr::client::read_response_bytes(response, limit),
        )
        .await;
        server.abort();
        let _ = server.await;
        result.expect("bounded reads must finish without waiting for the rest of an oversized body")
    }

    #[tokio::test]
    async fn response_limit_accepts_exact_size_and_rejects_declared_and_chunked_overflow() {
        for response in [
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nabcdefgh",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n4\r\nefgh\r\n0\r\n\r\n",
        ] {
            assert_eq!(
                read_bounded_response(response.as_bytes().to_vec(), 8)
                    .await
                    .unwrap(),
                "abcdefgh"
            );
        }
        for response in [
            "HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n5\r\nefghi\r\n",
        ] {
            assert!(matches!(
                read_bounded_response(response.as_bytes().to_vec(), 8).await,
                Err(crate::ocr::Error::TooLarge { limit: 8 })
            ));
        }
    }

    #[tokio::test]
    async fn oversized_error_retains_http_status_and_bounded_diagnostics_without_draining() {
        let prefix = "x".repeat(4 * (crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS + 1));
        for headers in ["Content-Length: 1000000", "Transfer-Encoding: chunked"] {
            let body = if headers.starts_with("Transfer") {
                format!("{:x}\r\n{prefix}\r\n", prefix.len())
            } else {
                prefix.clone()
            };
            let response = format!("HTTP/1.1 429 Too Many Requests\r\n{headers}\r\n\r\n{body}");
            let error = read_bounded_response(response.into_bytes(), 4096)
                .await
                .unwrap_err();
            match error {
                crate::ocr::Error::Transport(crate::transport::Error::Http { status, body }) => {
                    assert_eq!(status, 429);
                    assert_eq!(
                        body,
                        format!(
                            "{}... (truncated)",
                            "x".repeat(crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS)
                        )
                    );
                }
                error => panic!("unexpected error: {error}"),
            }
        }
    }

    #[test]
    fn response_limit_is_validated_and_not_forwarded_to_the_provider() {
        let request = wire_request(
            "mistral/model",
            "http://localhost",
            json!({"max_response_bytes": 123}),
        );
        assert_eq!(request.connection.max_response_bytes, 123);
        assert!(!request.optional_params.contains_key("max_response_bytes"));
        for value in [
            json!(0),
            json!(-1),
            json!(true),
            json!("123"),
            json!(1.5),
            json!(crate::constants::OCR_RESPONSE_MAX_BYTES + 1),
            Value::Null,
        ] {
            let wire = serde_json::from_value(json!({
                "model": "mistral/model", "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                "optional_params": {"max_response_bytes": value}
            })).unwrap();
            let Err(error) = decode_request(wire) else {
                panic!("invalid response limit accepted")
            };
            assert!(error.to_string().contains("max_response_bytes"));
        }
    }

    #[derive(Debug)]
    struct PendingToken {
        entered: Arc<tokio::sync::Notify>,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    struct TokenFutureDrop(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for TokenFutureDrop {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl litellm_auth::TokenProvider for PendingToken {
        fn acquire(&self) -> litellm_auth::TokenFuture<'_> {
            Box::pin(async move {
                let _guard = TokenFutureDrop(self.dropped.clone());
                self.entered.notify_one();
                std::future::pending().await
            })
        }
    }

    #[tokio::test]
    async fn cancellation_waits_for_provider_capture_drop_even_when_acknowledgement_is_cancelled() {
        use crate::call_lifecycle::host::HostFailure;
        use std::future::Future;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::task::Poll;

        for interrupt_acknowledgement in [false, true] {
            let entered = Arc::new(tokio::sync::Notify::new());
            let dropped = Arc::new(AtomicBool::new(false));
            let request =
                wire_request("azure_ai/mistral-ocr", "https://example.invalid", json!({}));
            let request = crate::ocr::LiteLLMOcrRequest {
                connection: crate::ocr::OcrConnection {
                    extra_headers: vec![("authorization".into(), "Bearer test-key".into())],
                    ..request.connection
                },
                azure_ad_token_provider: Some(litellm_auth::TokenProviderHandle::new(Arc::new(
                    PendingToken {
                        entered: entered.clone(),
                        dropped: dropped.clone(),
                    },
                ))),
                ..request
            };
            let NativeOutcome::Completed(mut call) =
                OcrCall::admit(crate::ocr::test_support::ocr_client(), OcrAdmission::all())
            else {
                panic!("supported call declined")
            };
            let mut request = Some(request);
            let mut result = None;
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                loop {
                    tokio::select! {
                        _ = entered.notified() => break,
                        step = call.resume(result.take()) => {
                            result = Some(match step.unwrap() {
                                OcrCallStep::Host(OcrHostOperation::ProjectRequest) => OcrHostResult::Request(Ok((Box::new(request.take().unwrap()), false))),
                                OcrCallStep::Host(operation) => NoopOcrHost.invoke(operation).await,
                                OcrCallStep::Complete(_) => panic!("pending provider completed"),
                            });
                        }
                    }
                }
            }).await.unwrap();
            assert!(!dropped.load(Ordering::SeqCst));
            let selected = crate::ocr::Error::InvalidRequest("cancelled".into());
            if interrupt_acknowledgement {
                let mut acknowledgement =
                    Box::pin(call.interrupt(HostFailure::Cancelled(selected.clone())));
                std::future::poll_fn(|cx| {
                    assert!(acknowledgement.as_mut().poll(cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                drop(acknowledgement);
                assert!(!dropped.load(Ordering::SeqCst));
            }
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                call.interrupt(HostFailure::Cancelled(selected.clone())),
            )
            .await
            .unwrap();
            assert!(matches!(result, Err(error) if error == selected));
            assert!(
                dropped.load(Ordering::SeqCst),
                "cancellation returned while provider captures were still alive"
            );
        }
    }
}
