use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use litellm_python_extension_protocol::python_extension_host_client::PythonExtensionHostClient;
use litellm_python_extension_protocol::{
    CallbackEvent, CommitRevisionRequest, ErrorCode, ExtensionDescriptor, GetCapabilitiesRequest,
    GuardrailDecision, GuardrailInvocation, GuardrailResult, PrepareRevisionRequest,
    PublishCallbackEventsRequest, RetireRevisionRequest, StreamFrame,
};
use tokio::sync::mpsc;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::service::Interceptor;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Request, Status, Streaming};

use super::config::ManifestExtensionKind;
use super::config::{PythonExtensionManifest, PythonExtensionSettings};

const PROTOCOL_MAJOR: u32 = 1;
const PROTOCOL_MINOR: u32 = 0;
const TOKEN_METADATA_KEY: &str = "x-litellm-extension-token";

type HostStub = PythonExtensionHostClient<InterceptedService<Channel, TokenInterceptor>>;

#[derive(Clone)]
struct TokenInterceptor {
    token: MetadataValue<Ascii>,
}

impl Interceptor for TokenInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert(TOKEN_METADATA_KEY, self.token.clone());
        Ok(request)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionHostHealth {
    pub healthy: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivationState {
    Active(Vec<ExtensionDescriptor>),
    Degraded(String),
}

#[derive(Debug)]
pub enum InitializationError {
    InvalidConfiguration(String),
    Rejected(String),
}

impl std::fmt::Display for InitializationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) | Self::Rejected(message) => {
                formatter.write_str(message)
            }
        }
    }
}

impl std::error::Error for InitializationError {}

#[derive(Clone)]
pub struct PythonExtensionClient {
    stub: HostStub,
    settings: PythonExtensionSettings,
    manifest: PythonExtensionManifest,
    callback_tx: mpsc::Sender<CallbackEvent>,
    health: Arc<RwLock<ExtensionHostHealth>>,
    bypass_counts: Arc<Mutex<HashMap<(String, String, String), u64>>>,
    recovering: Arc<AtomicBool>,
}

impl PythonExtensionClient {
    pub async fn connect(
        settings: PythonExtensionSettings,
        manifest: PythonExtensionManifest,
    ) -> Result<(Arc<Self>, ActivationState), InitializationError> {
        let token = MetadataValue::try_from(settings.token.as_str()).map_err(|error| {
            InitializationError::InvalidConfiguration(format!("invalid extension token: {error}"))
        })?;
        let endpoint = Endpoint::from_shared(settings.endpoint.clone()).map_err(|error| {
            InitializationError::InvalidConfiguration(format!(
                "invalid extension endpoint: {error}"
            ))
        })?;
        let channel = endpoint
            .connect_timeout(settings.connect_timeout)
            .connect_lazy();
        let stub = PythonExtensionHostClient::with_interceptor(channel, TokenInterceptor { token });
        let (callback_tx, callback_rx) = mpsc::channel(settings.callback_queue_size);
        let client = Arc::new(Self {
            stub,
            settings,
            manifest,
            callback_tx,
            health: Arc::new(RwLock::new(ExtensionHostHealth {
                healthy: false,
                reason: Some("not connected".to_string()),
            })),
            bypass_counts: Arc::new(Mutex::new(HashMap::new())),
            recovering: Arc::new(AtomicBool::new(false)),
        });
        tokio::spawn(client.clone().callback_worker(callback_rx));
        let activation = match client.activate().await {
            Ok(descriptors) => ActivationState::Active(descriptors),
            Err(status) if is_transient(&status) => {
                let reason = status.code().to_string();
                client.mark_unhealthy(reason.clone());
                client.schedule_recovery();
                ActivationState::Degraded(reason)
            }
            Err(status) => {
                return Err(InitializationError::Rejected(format!(
                    "extension host rejected startup: {}",
                    status.message()
                )));
            }
        };
        Ok((client, activation))
    }

    pub fn manifest(&self) -> &PythonExtensionManifest {
        &self.manifest
    }

    pub fn health(&self) -> ExtensionHostHealth {
        self.health
            .read()
            .map(|health| health.clone())
            .unwrap_or(ExtensionHostHealth {
                healthy: false,
                reason: Some("health lock poisoned".to_string()),
            })
    }

    pub fn bypass_counts(&self) -> HashMap<(String, String, String), u64> {
        self.bypass_counts
            .lock()
            .map(|counts| counts.clone())
            .unwrap_or_default()
    }

    pub async fn execute_guardrail(&self, invocation: GuardrailInvocation) -> GuardrailResult {
        let plugin_id = invocation.plugin_id.clone();
        let hook = invocation.hook_phase.to_string();
        let mut request = Request::new(invocation);
        request.set_timeout(self.settings.hook_timeout);
        let mut stub = self.stub.clone();
        match stub.execute_guardrail(request).await {
            Ok(response) => {
                let result = response.into_inner();
                if let Some(operation) = result.operation.as_ref().filter(|operation| !operation.ok)
                {
                    self.record_bypass(&plugin_id, &hook, &operation_reason(operation));
                } else {
                    self.mark_healthy();
                }
                result
            }
            Err(status) => {
                self.record_bypass(&plugin_id, &hook, status.code().description());
                self.schedule_recovery();
                GuardrailResult {
                    operation: Some(litellm_python_extension_protocol::OperationResult {
                        ok: true,
                        ..Default::default()
                    }),
                    decision: GuardrailDecision::Allow.into(),
                    ..Default::default()
                }
            }
        }
    }

    pub fn enqueue_callback(&self, event: CallbackEvent) -> Result<(), &'static str> {
        let plugin_id = event.plugin_id.clone();
        match self.callback_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.record_bypass(&plugin_id, "callback", "queue_full");
                Err("callback queue is full")
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.record_bypass(&plugin_id, "callback", "queue_closed");
                Err("callback queue is closed")
            }
        }
    }

    pub(crate) fn record_stream_bypass(&self, reason: &str) {
        self.record_bypass("stream", "transform", reason);
        self.schedule_recovery();
    }

    pub async fn transform_stream<S>(&self, frames: S) -> Result<Streaming<StreamFrame>, Status>
    where
        S: futures_util::Stream<Item = StreamFrame> + Send + 'static,
    {
        let mut request = Request::new(frames);
        request.set_timeout(self.settings.hook_timeout);
        let mut stub = self.stub.clone();
        stub.transform_stream(request)
            .await
            .map(tonic::Response::into_inner)
    }

    pub async fn retire(&self, revision_id: String) {
        let mut request = Request::new(RetireRevisionRequest { revision_id });
        request.set_timeout(self.settings.hook_timeout);
        let mut stub = self.stub.clone();
        let _ = stub.retire_revision(request).await;
    }

    async fn activate(&self) -> Result<Vec<ExtensionDescriptor>, Status> {
        let mut capabilities_request = Request::new(GetCapabilitiesRequest {
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor: PROTOCOL_MINOR,
        });
        capabilities_request.set_timeout(self.settings.connect_timeout);
        let mut stub = self.stub.clone();
        let capabilities = stub
            .get_capabilities(capabilities_request)
            .await?
            .into_inner();
        if capabilities.protocol_major != PROTOCOL_MAJOR {
            return Err(Status::failed_precondition(format!(
                "protocol major {} does not match {PROTOCOL_MAJOR}",
                capabilities.protocol_major
            )));
        }
        let has_callbacks = self
            .manifest
            .extensions
            .iter()
            .any(|extension| extension.kind == ManifestExtensionKind::Callback);
        if has_callbacks && !capabilities.supports_callback_batching {
            return Err(Status::failed_precondition(
                "extension host does not support callback batching",
            ));
        }
        if capabilities.max_callback_batch_size > 0
            && self.settings.callback_batch_size > capabilities.max_callback_batch_size as usize
        {
            return Err(Status::failed_precondition(
                "callback batch size exceeds extension host capability",
            ));
        }
        let extensions = self
            .manifest
            .specs()
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let mut prepare_request = Request::new(PrepareRevisionRequest {
            revision_id: self.manifest.revision_id.clone(),
            extensions,
        });
        prepare_request.set_timeout(self.settings.hook_timeout);
        let prepared = stub.prepare_revision(prepare_request).await?.into_inner();
        let operation = prepared
            .operation
            .ok_or_else(|| Status::internal("PrepareRevision omitted operation"))?;
        if !operation.ok && operation.error_code != ErrorCode::AlreadyExists as i32 {
            return Err(Status::failed_precondition(operation.error_message));
        }
        if !capabilities.supports_duplex_streaming
            && prepared.extensions.iter().any(|descriptor| {
                descriptor.hooks.iter().any(|hook| {
                    matches!(
                        hook.as_str(),
                        "async_post_call_streaming_hook"
                            | "async_post_call_streaming_iterator_hook"
                    )
                })
            })
        {
            return Err(Status::failed_precondition(
                "extension host does not support required duplex streaming hooks",
            ));
        }
        let mut commit_request = Request::new(CommitRevisionRequest {
            revision_id: self.manifest.revision_id.clone(),
        });
        commit_request.set_timeout(self.settings.hook_timeout);
        let committed = stub.commit_revision(commit_request).await?.into_inner();
        if !committed.ok {
            return Err(Status::failed_precondition(committed.error_message));
        }
        self.mark_healthy();
        Ok(prepared.extensions)
    }

    async fn callback_worker(self: Arc<Self>, mut receiver: mpsc::Receiver<CallbackEvent>) {
        while let Some(first) = receiver.recv().await {
            let mut events = vec![first];
            while events.len() < self.settings.callback_batch_size {
                match receiver.try_recv() {
                    Ok(event) => events.push(event),
                    Err(_) => break,
                }
            }
            let mut request = Request::new(PublishCallbackEventsRequest {
                events: events.clone(),
            });
            request.set_timeout(self.settings.hook_timeout);
            let mut stub = self.stub.clone();
            match stub.publish_callback_events(request).await {
                Ok(response) => {
                    let operations = response.into_inner().operations;
                    let mut all_ok = operations.len() == events.len();
                    for (index, event) in events.iter().enumerate() {
                        match operations.get(index) {
                            Some(operation) if operation.ok => {}
                            Some(operation) => {
                                all_ok = false;
                                self.record_bypass(
                                    &event.plugin_id,
                                    "callback",
                                    &operation_reason(operation),
                                );
                            }
                            None => {
                                all_ok = false;
                                self.record_bypass(
                                    &event.plugin_id,
                                    "callback",
                                    "missing_operation",
                                );
                            }
                        }
                    }
                    if all_ok {
                        self.mark_healthy();
                    }
                }
                Err(status) => {
                    for event in events {
                        self.record_bypass(
                            &event.plugin_id,
                            "callback",
                            status.code().description(),
                        );
                    }
                    self.schedule_recovery();
                }
            }
        }
    }

    fn schedule_recovery(&self) {
        if self.recovering.swap(true, Ordering::AcqRel) {
            return;
        }
        let client = self.clone();
        tokio::spawn(async move {
            let mut delay = Duration::from_millis(250);
            loop {
                match client.activate().await {
                    Ok(_) => break,
                    Err(status) => {
                        client.mark_unhealthy(status.code().to_string());
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(Duration::from_secs(5));
                    }
                }
            }
            client.recovering.store(false, Ordering::Release);
        });
    }

    fn record_bypass(&self, plugin_id: &str, hook: &str, reason: &str) {
        if let Ok(mut counts) = self.bypass_counts.lock() {
            *counts
                .entry((plugin_id.to_string(), hook.to_string(), reason.to_string()))
                .or_insert(0) += 1;
        }
        self.mark_unhealthy(reason.to_string());
        eprintln!("python_extension_host_bypass plugin={plugin_id} hook={hook} reason={reason}");
    }

    fn mark_healthy(&self) {
        if let Ok(mut health) = self.health.write() {
            *health = ExtensionHostHealth {
                healthy: true,
                reason: None,
            };
        }
    }

    fn mark_unhealthy(&self, reason: String) {
        if let Ok(mut health) = self.health.write() {
            *health = ExtensionHostHealth {
                healthy: false,
                reason: Some(reason),
            };
        }
    }
}

fn is_transient(status: &Status) -> bool {
    matches!(
        status.code(),
        Code::Unavailable | Code::DeadlineExceeded | Code::Cancelled | Code::Unknown
    )
}

fn operation_reason(operation: &litellm_python_extension_protocol::OperationResult) -> String {
    let code = ErrorCode::try_from(operation.error_code).unwrap_or(ErrorCode::Unspecified);
    if operation.error_message.is_empty() {
        format!("{code:?}")
    } else {
        format!("{code:?}:{}", operation.error_message)
    }
}
