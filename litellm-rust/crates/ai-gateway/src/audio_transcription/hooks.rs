use litellm_core::CoreResult;
use litellm_core::call_lifecycle::{
    CallContext, CallInterceptor, CallOutcome, CallSpec, CallTiming,
};
use litellm_core::callbacks::CallbackOptions;
use litellm_core::callbacks::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailRequest,
};
use litellm_core::callbacks::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use litellm_core::callbacks::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload,
};
use litellm_core::error::CoreError;
use serde_json::{Map, Value, json};

use super::types::{
    AudioTranscriptionCall, PreparedAudioTranscriptionRequest, ProviderAudioTranscriptionRequest,
};

pub(crate) struct AudioTranscriptionCallbackInterceptor {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

impl AudioTranscriptionCallbackInterceptor {
    pub(crate) fn new(options: CallbackOptions) -> Self {
        Self {
            logger_runner: CustomLoggerRunner::new(options.callbacks),
            guardrail_runner: CustomGuardrailRunner::new(options.guardrails),
            request_metadata: options.request_metadata,
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
            .await?;
        let Value::Object(mut data) = guardrail_request.data else {
            return Err(CoreError::invalid_request(
                "audio transcription pre_call guardrail must return an object".to_string(),
            ));
        };
        let audio = data.remove("audio").ok_or_else(|| {
            CoreError::invalid_request("audio transcription guardrail removed audio".to_string())
        })?;
        let optional_params = match data.remove("optional_params") {
            Some(Value::Object(value)) => value,
            Some(_) => {
                return Err(CoreError::invalid_request(
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

    async fn run_before_send_guardrails(
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
                    "model": request.model,
                    "custom_llm_provider": request.custom_llm_provider,
                    "url": request.url,
                    "body": request.body,
                })),
            )
            .await?;
        let Value::Object(mut data) = guardrail_request.data else {
            return Err(CoreError::invalid_request(
                "audio transcription before_send guardrail must return an object".to_string(),
            ));
        };
        let body = data.remove("body").ok_or_else(|| {
            CoreError::invalid_request("audio transcription guardrail removed body".to_string())
        })?;
        Ok(ProviderAudioTranscriptionRequest { body, ..request })
    }

    fn logging_payload(
        &self,
        context: &CallContext,
        timing: &CallTiming,
    ) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: context.litellm_call_id.clone(),
            litellm_call_id: context.litellm_call_id.clone(),
            call_type: AudioTranscriptionCall::NAME.to_string(),
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

impl CallInterceptor<AudioTranscriptionCall> for AudioTranscriptionCallbackInterceptor {
    async fn before_call<'a>(
        &'a self,
        _context: &'a CallContext,
        input: PreparedAudioTranscriptionRequest,
    ) -> CoreResult<PreparedAudioTranscriptionRequest> {
        self.run_pre_call_guardrails(input).await
    }

    async fn before_send<'a>(
        &'a self,
        _context: &'a CallContext,
        input: ProviderAudioTranscriptionRequest,
    ) -> CoreResult<ProviderAudioTranscriptionRequest> {
        self.run_before_send_guardrails(input).await
    }

    async fn complete<'a>(
        &'a self,
        context: &'a CallContext,
        outcome: CallOutcome<'a, Value>,
        timing: &'a CallTiming,
    ) {
        if self.logger_runner.is_empty() {
            return;
        }

        match outcome {
            CallOutcome::Success(response) => {
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
            CallOutcome::Failure(error) => {
                let logging_error = LoggingError {
                    message: error.to_string(),
                    kind: error.kind().to_string(),
                };
                self.logger_runner
                    .async_log_failure_event(
                        &ModelCallDetails::from_standard_logging_payload(
                            self.logging_payload(context, timing),
                        )
                        .with_failure_error(logging_error.clone()),
                        Some(&CallbackValue::new(
                            "error",
                            json!({
                                "message": logging_error.message,
                                "kind": logging_error.kind,
                            }),
                        )),
                        CallbackTiming::new(timing.start_time, timing.end_time),
                    )
                    .await;
            }
        }
    }
}

fn guardrail_context(metadata: &RequestMetadata) -> GuardrailContext {
    GuardrailContext {
        call_type: CallType::Other(AudioTranscriptionCall::NAME.to_string()),
        selected_guardrails: Vec::new(),
        metadata: std::collections::HashMap::new(),
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        trace_parent: None,
    }
}
