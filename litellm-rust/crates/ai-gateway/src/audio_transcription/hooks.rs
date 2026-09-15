use serde_json::{Map, Value, json};

use litellm_core::call_lifecycle::provider::{
    ProviderHookFuture, ProviderHooks, ProviderRequest, ProviderResponse,
};
use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
use litellm_core::error::Error;

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
    provider: String,
}

impl AudioTranscriptionLifecycleHooks {
    pub(crate) fn new(
        logger_runner: CustomLoggerRunner,
        guardrail_runner: CustomGuardrailRunner,
        request_metadata: RequestMetadata,
        provider: String,
    ) -> Self {
        Self {
            logger_runner,
            guardrail_runner,
            request_metadata,
            provider,
        }
    }

    pub(crate) async fn run_pre_call_guardrails(
        &self,
        request: PreparedAudioTranscriptionRequest,
    ) -> Result<PreparedAudioTranscriptionRequest, Error> {
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
            return Err(Error::InvalidRequest(
                "audio transcription pre_call guardrail must return an object".to_string(),
            ));
        };
        let audio = data.remove("audio").ok_or_else(|| {
            Error::InvalidRequest("audio transcription guardrail removed audio".to_string())
        })?;
        let optional_params = match data.remove("optional_params") {
            Some(Value::Object(value)) => value,
            Some(_) => {
                return Err(Error::InvalidRequest(
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

    async fn run_during_call_guardrails(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderRequest, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_during_call(
                &guardrail_context(&self.request_metadata),
                GuardrailRequest::new(json!({
                    "model": &request.model,
                    "custom_llm_provider": &self.provider,
                    "url": &request.url,
                    "body": &request.body,
                })),
            )
            .await
            .map_err(guardrail_error_to_core_error)?;
        let Value::Object(mut data) = guardrail_request.data else {
            return Err(Error::InvalidRequest(
                "audio transcription during_call guardrail must return an object".to_string(),
            ));
        };
        let body = data.remove("body").ok_or_else(|| {
            Error::InvalidRequest("audio transcription guardrail removed body".to_string())
        })?;
        Ok(ProviderRequest { body, ..request })
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
    pub(crate) async fn log_success(
        &self,
        context: &CallLifecycleContext,
        response: &Value,
        timing: &CallLifecycleTiming,
    ) {
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
    }

    pub(crate) async fn log_failure(
        &self,
        context: &CallLifecycleContext,
        error: &Error,
        timing: &CallLifecycleTiming,
    ) {
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
    }
}

impl ProviderHooks for AudioTranscriptionLifecycleHooks {
    fn before_request(&self, request: ProviderRequest) -> ProviderHookFuture<'_, ProviderRequest> {
        Box::pin(async move { self.run_during_call_guardrails(request).await })
    }

    fn after_response(
        &self,
        response: ProviderResponse,
    ) -> ProviderHookFuture<'_, ProviderResponse> {
        Box::pin(async move { Ok(response) })
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

fn guardrail_error_to_core_error(error: GuardrailError) -> Error {
    Error::InvalidRequest(format!("{}: {}", error.kind, error.message))
}

fn core_error_kind(error: &Error) -> &'static str {
    match error {
        Error::Auth(_)
        | Error::MissingApiKey { .. }
        | Error::MissingAzureAiCredentials
        | Error::MissingAzureDocumentIntelligenceCredentials
        | Error::MissingReductoApiKey => "AuthError",
        Error::InvalidProvider(_) => "InvalidProvider",
        Error::InvalidRequest(_) => "InvalidRequest",
        Error::InvalidType { .. } => "InvalidType",
        Error::MissingField(_) | Error::MissingDocumentUrl => "MissingField",
        Error::Http { .. } => "HttpError",
        Error::InvalidResponse(_) => "InvalidResponse",
        Error::Network(_) => "NetworkError",
        Error::Connect(_) => "ConnectError",
        Error::Routing(_) => "RoutingError",
        Error::Unsupported(_) => "UnsupportedRequest",
    }
}
