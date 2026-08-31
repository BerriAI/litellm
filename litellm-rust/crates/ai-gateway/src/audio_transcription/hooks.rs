use std::future::Future;
use std::pin::Pin;

use litellm_core::CoreResult;
use litellm_core::audio_transcription::{
    AudioTranscriptionRequest as CoreAudioTranscriptionRequest, ProviderAudioTranscriptionRequest,
    prepare_audio_transcription_provider_call,
};
use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming};
use litellm_core::error::CoreError;
use serde_json::{Map, Value, json};

use super::types::PreparedAudioTranscriptionRequest;
use crate::integrations::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload,
};

pub(crate) struct AudioTranscriptionLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

type AudioFuture<'a, T> = Pin<Box<dyn Future<Output = CoreResult<T>> + Send + 'a>>;
type AudioLogFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

impl AudioTranscriptionLifecycleHooks {
    pub(crate) fn new(
        logger_runner: CustomLoggerRunner,
        guardrail_runner: CustomGuardrailRunner,
        request_metadata: RequestMetadata,
    ) -> Self {
        Self {
            logger_runner,
            guardrail_runner,
            request_metadata,
        }
    }

    async fn run_pre_call_guardrails(
        &self,
        request: PreparedAudioTranscriptionRequest,
    ) -> CoreResult<PreparedAudioTranscriptionRequest> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_pre_call(
                &guardrail_context(&self.request_metadata),
                GuardrailRequest::new(json!({
                    "model": request.model,
                    "custom_llm_provider": request.custom_llm_provider,
                    "audio": request.audio,
                    "optional_params": request.optional_params,
                })),
            )
            .await
            .map_err(guardrail_error_to_core_error)?;
        let Value::Object(mut data) = guardrail_request.data else {
            return Err(CoreError::InvalidRequest(
                "audio transcription pre_call guardrail must return an object".to_string(),
            ));
        };
        let audio = data.remove("audio").ok_or_else(|| {
            CoreError::InvalidRequest("audio transcription guardrail removed audio".to_string())
        })?;
        let optional_params = match data.remove("optional_params") {
            Some(Value::Object(value)) => value,
            Some(_) => {
                return Err(CoreError::InvalidRequest(
                    "audio transcription optional_params must be an object".to_string(),
                ));
            }
            None => Map::new(),
        };
        Ok(PreparedAudioTranscriptionRequest {
            audio,
            optional_params,
            ..request
        })
    }

    async fn prepare_provider_request(
        &self,
        request: PreparedAudioTranscriptionRequest,
    ) -> CoreResult<ProviderAudioTranscriptionRequest> {
        let PreparedAudioTranscriptionRequest {
            model,
            custom_llm_provider,
            audio,
            api_key,
            api_base,
            extra_headers,
            optional_params,
            timeout,
            ..
        } = request;
        let provider_request =
            prepare_audio_transcription_provider_call(CoreAudioTranscriptionRequest {
                model: &model,
                audio,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: Some(&custom_llm_provider),
                extra_headers,
                optional_params,
                timeout,
            })?;
        self.run_during_call_guardrails(provider_request).await
    }

    async fn run_during_call_guardrails(
        &self,
        request: ProviderAudioTranscriptionRequest,
    ) -> CoreResult<ProviderAudioTranscriptionRequest> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_during_call(
                &guardrail_context(&self.request_metadata),
                GuardrailRequest::new(json!({
                    "model": request.model(),
                    "custom_llm_provider": request.custom_llm_provider(),
                    "url": request.url(),
                    "body": request.body(),
                })),
            )
            .await
            .map_err(guardrail_error_to_core_error)?;
        let Value::Object(mut data) = guardrail_request.data else {
            return Err(CoreError::InvalidRequest(
                "audio transcription during_call guardrail must return an object".to_string(),
            ));
        };
        let body = data.remove("body").ok_or_else(|| {
            CoreError::InvalidRequest("audio transcription guardrail removed body".to_string())
        })?;
        Ok(request.with_body(body))
    }

    fn logging_payload(
        &self,
        context: &CallLifecycleContext,
        timing: &CallLifecycleTiming,
    ) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: context.litellm_call_id.clone(),
            litellm_call_id: context.litellm_call_id.clone(),
            call_type: context.call_type.clone(),
            model: context.model.clone(),
            custom_llm_provider: context.custom_llm_provider.clone(),
            response_cost: 0.0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            start_time: timing.start_time,
            end_time: timing.end_time,
            stream: false,
            metadata: StandardLoggingMetadata {
                user_api_key_hash: self.request_metadata.user_api_key_hash.clone(),
                user_api_key_user_id: self.request_metadata.user_api_key_user_id.clone(),
                user_api_key_team_id: self.request_metadata.user_api_key_team_id.clone(),
                ..Default::default()
            },
            messages: None,
        }
    }
}

impl CallLifecycleHooks<PreparedAudioTranscriptionRequest, ProviderAudioTranscriptionRequest, Value>
    for AudioTranscriptionLifecycleHooks
{
    type PreCallFuture<'a> = AudioFuture<'a, PreparedAudioTranscriptionRequest>;
    type DuringCallFuture<'a> = AudioFuture<'a, ProviderAudioTranscriptionRequest>;
    type SuccessFuture<'a> = AudioLogFuture<'a>;
    type FailureFuture<'a> = AudioLogFuture<'a>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedAudioTranscriptionRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move { self.run_pre_call_guardrails(request).await })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedAudioTranscriptionRequest,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { self.prepare_provider_request(request).await })
    }

    fn async_log_success_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: &'a Value,
        timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            self.logger_runner
                .async_log_success_event(
                    &ModelCallDetails::from_standard_logging_payload(
                        self.logging_payload(context, timing),
                    ),
                    &CallbackValue::new("audio_transcription", response.clone()),
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a CoreError,
        timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            let logging_error = LoggingError {
                message: error.to_string(),
                kind: core_error_kind(error).to_string(),
            };
            self.logger_runner
                .async_log_failure_event(
                    &ModelCallDetails::from_standard_logging_payload(
                        self.logging_payload(context, timing),
                    )
                    .with_failure_error(logging_error.clone()),
                    Some(&CallbackValue::new(
                        "error",
                        json!({"message": logging_error.message, "kind": logging_error.kind}),
                    )),
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }
}

fn guardrail_context(metadata: &RequestMetadata) -> GuardrailContext {
    GuardrailContext {
        call_type: CallType::Other("audio_transcription".to_string()),
        selected_guardrails: Vec::new(),
        metadata: std::collections::HashMap::new(),
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        trace_parent: None,
    }
}

fn guardrail_error_to_core_error(error: GuardrailError) -> CoreError {
    CoreError::InvalidRequest(format!("{}: {}", error.kind, error.message))
}

fn core_error_kind(error: &CoreError) -> &'static str {
    match error {
        CoreError::Auth(_) => "AuthError",
        CoreError::InvalidProvider(_) => "InvalidProvider",
        CoreError::InvalidRequest(_) => "InvalidRequest",
        CoreError::InvalidType { .. } => "InvalidType",
        CoreError::MissingField(_) => "MissingField",
        CoreError::Http { .. } => "HttpError",
        CoreError::InvalidResponse(_) => "InvalidResponse",
        CoreError::Network(_) => "NetworkError",
        CoreError::Connect(_) => "ConnectError",
        CoreError::Routing(_) => "RoutingError",
        CoreError::Unsupported(_) => "UnsupportedRequest",
    }
}
