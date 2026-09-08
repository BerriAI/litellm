use litellm_core::audio_transcription::{
    AudioTranscriptionRequest as CoreAudioTranscriptionRequest, ProviderAudioTranscriptionRequest,
    prepare_audio_transcription_provider_call,
};
use litellm_core::error::Error;
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, RequestPolicy, TerminalDispatcher, TerminalRecord,
};
use serde_json::{Map, Value, json};
use std::future::Future;
use std::pin::Pin;

use super::types::PreparedAudioTranscriptionRequest;
use litellm_core::integrations::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use litellm_core::integrations::custom_logger::{CallType, CustomLoggerRunner, LogFuture};
use litellm_core::integrations::types::RequestMetadata;

pub(crate) struct AudioTranscriptionLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

type AudioFuture<'a, T> = Pin<Box<dyn Future<Output = ActionResult<T, Error>> + Send + 'a>>;

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

    async fn prepare_provider_request(
        &self,
        request: PreparedAudioTranscriptionRequest,
    ) -> Result<ProviderAudioTranscriptionRequest, Error> {
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
    ) -> Result<ProviderAudioTranscriptionRequest, Error> {
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
            return Err(Error::InvalidRequest(
                "audio transcription during_call guardrail must return an object".to_string(),
            ));
        };
        let body = data.remove("body").ok_or_else(|| {
            Error::InvalidRequest("audio transcription guardrail removed body".to_string())
        })?;
        Ok(request.with_body(body))
    }
}

impl RequestPolicy<PreparedAudioTranscriptionRequest, ProviderAudioTranscriptionRequest>
    for AudioTranscriptionLifecycleHooks
{
    type PreCallFuture<'a> = AudioFuture<'a, PreparedAudioTranscriptionRequest>;
    type DuringCallFuture<'a> = AudioFuture<'a, ProviderAudioTranscriptionRequest>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedAudioTranscriptionRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move {
            match self.run_pre_call_guardrails(request).await {
                Ok(request) => ActionResult::Replace(request),
                Err(error) => ActionResult::Reject(error),
            }
        })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedAudioTranscriptionRequest,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move {
            match self.prepare_provider_request(request).await {
                Ok(request) => ActionResult::Replace(request),
                Err(error) => ActionResult::Reject(error),
            }
        })
    }
}

impl TerminalDispatcher for AudioTranscriptionLifecycleHooks {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        let mut terminal = terminal.clone();
        terminal.cost_inputs.metadata = request_metadata(&self.request_metadata);
        Box::pin(async move { self.logger_runner.dispatch(&terminal).await })
    }
}

fn request_metadata(
    metadata: &RequestMetadata,
) -> litellm_core::integrations::types::StandardLoggingMetadata {
    litellm_core::integrations::types::StandardLoggingMetadata {
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        ..Default::default()
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
