use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_python_extension_protocol::{
    AuthContext, CallbackEvent, CallbackEventKind, GuardrailDecision as WireDecision,
    GuardrailInvocation, HookPhase, InvocationContext, OperationResult,
};
use serde_json::{Value, json};

use crate::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailError, GuardrailEventHook,
    GuardrailFuture, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogError, LogFuture, ModelCallDetails,
};

use super::client::PythonExtensionClient;
use super::config::{ManifestExtensionKind, PythonExtensionManifest};

pub struct RemoteCustomGuardrail {
    name: String,
    plugin_id: String,
    hooks: Vec<GuardrailEventHook>,
    client: Arc<PythonExtensionClient>,
}

impl RemoteCustomGuardrail {
    pub fn new(
        name: String,
        plugin_id: String,
        hooks: Vec<GuardrailEventHook>,
        client: Arc<PythonExtensionClient>,
    ) -> Self {
        Self {
            name,
            plugin_id,
            hooks,
            client,
        }
    }

    fn invoke<'a>(
        &'a self,
        phase: HookPhase,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            let original = request.clone();
            let encoded = serde_json::to_vec(&request.data).map_err(|error| GuardrailError {
                message: error.to_string(),
                kind: "SerializationError".to_string(),
            })?;
            let (request_json, response_json) = if phase == HookPhase::PostCall {
                (b"{}".to_vec(), Some(encoded))
            } else {
                (encoded, None)
            };
            let result = self
                .client
                .execute_guardrail(GuardrailInvocation {
                    context: Some(invocation_context(
                        self.client.manifest().revision_id.clone(),
                        context.call_type.as_str(),
                    )),
                    plugin_id: self.plugin_id.clone(),
                    hook_phase: phase.into(),
                    request_json,
                    response_json,
                    auth: Some(auth_context(context)),
                    cache: None,
                })
                .await;
            wire_result_to_decision(
                result.operation,
                result.decision,
                result.request_json,
                result.response_json,
                result.public_error,
                original,
            )
        })
    }
}

impl CustomGuardrail for RemoteCustomGuardrail {
    fn guardrail_name(&self) -> &str {
        &self.name
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        &self.hooks
    }

    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.invoke(HookPhase::PreCall, context, request)
    }

    fn async_moderation_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.invoke(HookPhase::DuringCall, context, request)
    }

    fn async_post_call_success_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        response: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.invoke(HookPhase::PostCall, context, response)
    }
}

pub struct RemoteCustomLogger {
    plugin_id: String,
    success_enabled: bool,
    failure_enabled: bool,
    client: Arc<PythonExtensionClient>,
}

impl RemoteCustomLogger {
    pub fn new(plugin_id: String, client: Arc<PythonExtensionClient>) -> Self {
        Self::with_events(plugin_id, true, true, client)
    }

    fn with_events(
        plugin_id: String,
        success_enabled: bool,
        failure_enabled: bool,
        client: Arc<PythonExtensionClient>,
    ) -> Self {
        Self {
            plugin_id,
            success_enabled,
            failure_enabled,
            client,
        }
    }

    fn enqueue(
        &self,
        kind: CallbackEventKind,
        details: &ModelCallDetails,
        response: Option<&CallbackValue>,
        timing: CallbackTiming,
    ) -> Result<(), LogError> {
        let payload_json = details
            .standard_logging_payload
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| LogError {
                message: error.to_string(),
                kind: "SerializationError".to_string(),
            })?
            .unwrap_or_else(|| {
                serde_json::to_vec(&json!({
                    "model": details.model,
                    "custom_llm_provider": details.custom_llm_provider,
                    "call_type": details.call_type.as_str(),
                }))
                .unwrap_or_else(|_| b"{}".to_vec())
            });
        let response_json = response
            .map(|response| serde_json::to_vec(&response.value))
            .transpose()
            .map_err(|error| LogError {
                message: error.to_string(),
                kind: "SerializationError".to_string(),
            })?;
        let error_json = details.failure_error.as_ref().map(|error| {
            serde_json::to_vec(&json!({"type": error.kind, "message": error.message}))
                .unwrap_or_else(|_| b"{}".to_vec())
        });
        let metadata = &details.metadata;
        let event = CallbackEvent {
            context: Some(InvocationContext {
                request_id: details.request_id.clone().unwrap_or_default(),
                invocation_id: details
                    .litellm_call_id
                    .clone()
                    .unwrap_or_else(next_invocation_id),
                active_revision: self.client.manifest().revision_id.clone(),
                api_surface: details.call_type.as_str().to_string(),
                call_type: details.call_type.as_str().to_string(),
                trace_context: HashMap::new(),
            }),
            plugin_id: self.plugin_id.clone(),
            kind: kind.into(),
            standard_logging_payload_json: payload_json,
            response_json,
            error_json,
            start_time_seconds: timing.start_time,
            end_time_seconds: timing.end_time,
            auth: Some(AuthContext {
                key_hash: metadata.user_api_key_hash.clone().unwrap_or_default(),
                user_id: metadata.user_api_key_user_id.clone().unwrap_or_default(),
                team_id: metadata.user_api_key_team_id.clone().unwrap_or_default(),
                request_metadata: HashMap::new(),
            }),
            cache: None,
            streaming: details
                .standard_logging_payload
                .as_ref()
                .map(|payload| payload.stream)
                .unwrap_or(false),
        };
        self.client.enqueue_callback(event).map_err(|reason| {
            if reason.contains("full") {
                LogError::channel_full()
            } else {
                LogError::channel_closed()
            }
        })
    }
}

impl CustomLogger for RemoteCustomLogger {
    fn async_log_success_event<'a>(
        &'a self,
        details: &'a ModelCallDetails,
        response: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            if self.success_enabled {
                self.enqueue(CallbackEventKind::Success, details, Some(response), timing)
            } else {
                Ok(())
            }
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        details: &'a ModelCallDetails,
        response: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            if self.failure_enabled {
                self.enqueue(CallbackEventKind::Failure, details, response, timing)
            } else {
                Ok(())
            }
        })
    }
}

pub struct RemoteExtensions {
    pub guardrails: Vec<Arc<dyn CustomGuardrail>>,
    pub loggers: Vec<Arc<dyn CustomLogger>>,
}

impl RemoteExtensions {
    pub fn from_manifest(
        manifest: &PythonExtensionManifest,
        descriptors: &[litellm_python_extension_protocol::ExtensionDescriptor],
        client: Arc<PythonExtensionClient>,
    ) -> Self {
        let descriptor_hooks: HashMap<&str, &[String]> = descriptors
            .iter()
            .map(|descriptor| (descriptor.id.as_str(), descriptor.hooks.as_slice()))
            .collect();
        let mut guardrails: Vec<Arc<dyn CustomGuardrail>> = Vec::new();
        let mut loggers: Vec<Arc<dyn CustomLogger>> = Vec::new();
        for extension in &manifest.extensions {
            match extension.kind {
                ManifestExtensionKind::Callback => {
                    let events = extension
                        .constructor
                        .get("callback_events")
                        .and_then(Value::as_array);
                    let event_enabled = |name: &str| {
                        events.is_none_or(|events| {
                            events.iter().any(|event| event.as_str() == Some(name))
                        })
                    };
                    loggers.push(Arc::new(RemoteCustomLogger::with_events(
                        extension.id.clone(),
                        event_enabled("success"),
                        event_enabled("failure"),
                        client.clone(),
                    )));
                }
                ManifestExtensionKind::Guardrail => {
                    let name = extension
                        .constructor
                        .pointer("/kwargs/guardrail_name")
                        .and_then(Value::as_str)
                        .unwrap_or(&extension.id)
                        .to_string();
                    let hooks = descriptor_hooks
                        .get(extension.id.as_str())
                        .map(|hooks| hooks_from_descriptor(hooks))
                        .filter(|hooks| !hooks.is_empty())
                        .unwrap_or_else(|| {
                            vec![
                                GuardrailEventHook::PreCall,
                                GuardrailEventHook::DuringCall,
                                GuardrailEventHook::PostCall,
                            ]
                        });
                    guardrails.push(Arc::new(RemoteCustomGuardrail::new(
                        name,
                        extension.id.clone(),
                        hooks,
                        client.clone(),
                    )));
                }
            }
        }
        Self {
            guardrails,
            loggers,
        }
    }
}

fn wire_result_to_decision(
    operation: Option<OperationResult>,
    decision: i32,
    request_json: Option<Vec<u8>>,
    response_json: Option<Vec<u8>>,
    public_error: Option<litellm_python_extension_protocol::PublicError>,
    original: GuardrailRequest,
) -> Result<GuardrailDecision, GuardrailError> {
    if !operation.map(|operation| operation.ok).unwrap_or(false) {
        return Ok(GuardrailDecision::Allow(original));
    }
    match WireDecision::try_from(decision).unwrap_or(WireDecision::Error) {
        WireDecision::Allow | WireDecision::Error | WireDecision::Unspecified => {
            Ok(GuardrailDecision::Allow(original))
        }
        WireDecision::ReplaceRequest | WireDecision::ReplaceResponse => {
            let replacement = request_json
                .or(response_json)
                .ok_or_else(|| GuardrailError {
                    message: "extension replacement omitted JSON body".to_string(),
                    kind: "ExtensionProtocolError".to_string(),
                })?;
            let data = serde_json::from_slice(&replacement).map_err(|error| GuardrailError {
                message: error.to_string(),
                kind: "SerializationError".to_string(),
            })?;
            Ok(GuardrailDecision::Mask(GuardrailRequest::new(data)))
        }
        WireDecision::Block => Ok(GuardrailDecision::Block(GuardrailError::blocked(
            public_error
                .map(|error| error.message)
                .unwrap_or_else(|| "blocked by Python extension".to_string()),
        ))),
    }
}

fn invocation_context(revision_id: String, call_type: &str) -> InvocationContext {
    let invocation_id = next_invocation_id();
    InvocationContext {
        request_id: invocation_id.clone(),
        invocation_id,
        active_revision: revision_id,
        api_surface: call_type.to_string(),
        call_type: call_type.to_string(),
        trace_context: HashMap::new(),
    }
}

fn auth_context(context: &GuardrailContext) -> AuthContext {
    let request_metadata = context
        .metadata
        .iter()
        .filter(|(name, _)| !is_sensitive_name(name))
        .filter_map(|(name, value)| scalar_string(value).map(|value| (name.clone(), value)))
        .collect();
    AuthContext {
        key_hash: context.user_api_key_hash.clone().unwrap_or_default(),
        user_id: context.user_api_key_user_id.clone().unwrap_or_default(),
        team_id: context.user_api_key_team_id.clone().unwrap_or_default(),
        request_metadata,
    }
}

fn is_sensitive_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    [
        "authorization",
        "api_key",
        "token",
        "cookie",
        "secret",
        "password",
    ]
    .iter()
    .any(|part| name.contains(part))
}

fn scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn hooks_from_descriptor(hooks: &[String]) -> Vec<GuardrailEventHook> {
    hooks
        .iter()
        .filter_map(|hook| match hook.as_str() {
            "async_pre_call_hook" => Some(GuardrailEventHook::PreCall),
            "async_moderation_hook" => Some(GuardrailEventHook::DuringCall),
            "async_post_call_success_hook" => Some(GuardrailEventHook::PostCall),
            _ => None,
        })
        .collect()
}

pub(crate) fn next_invocation_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("extension-{timestamp}-{sequence}")
}
