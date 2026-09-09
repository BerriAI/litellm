use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::Error;
use crate::integrations::custom_guardrail::{
    CustomGuardrail, CustomGuardrailRunner, GuardrailContext, GuardrailError,
};
use crate::integrations::custom_logger::{CallType, CustomLogger, CustomLoggerRunner, LogFuture};
use crate::integrations::types::{RequestMetadata, StandardLoggingMetadata};
use crate::lifecycle::{
    ActionResult, CallLifecycle, CallLifecycleContext, Clock, DeploymentFailureHooks,
    DeploymentPreHooks, DeploymentSuccessHooks, ExecutedCall, ModerationHooks, PreCallHooks,
    TerminalDispatcher, TerminalRecord,
};
use crate::providers::dispatch::resolve_audio_route_provider;

use super::handler::execute_audio_transcription_provider_call;
use super::prepare::prepare_audio_transcription_provider_call;
use super::types::{
    AudioDuringCallGuardrailRequest, AudioInput, AudioPreCallGuardrailRequest, AudioRouteRequest,
    AudioTranscriptionRequest, ProviderAudioTranscriptionRequest,
};

pub trait AudioServices:
    TerminalDispatcher + Clock + crate::providers::auth::AuthorizationServices
{
    fn guardrails(&self) -> AudioGuardrailRunner;
}

pub type AudioGuardrail = dyn CustomGuardrail<
        PreCallRequest = AudioPreCallGuardrailRequest,
        DuringCallRequest = AudioDuringCallGuardrailRequest,
    >;
pub type AudioGuardrailRunner =
    CustomGuardrailRunner<AudioPreCallGuardrailRequest, AudioDuringCallGuardrailRequest>;

pub struct DefaultAudioServices {
    dispatcher: CustomLoggerRunner,
    guardrails: AudioGuardrailRunner,
    authorization: Arc<crate::providers::auth::NativeAuthorizationServices>,
}

impl DefaultAudioServices {
    pub fn new(
        callbacks: Vec<Arc<dyn CustomLogger>>,
        guardrails: Vec<Arc<AudioGuardrail>>,
    ) -> Self {
        Self::with_authorization(
            callbacks,
            guardrails,
            crate::providers::auth::shared_native_authorization_services().clone(),
        )
    }

    pub fn with_authorization(
        callbacks: Vec<Arc<dyn CustomLogger>>,
        guardrails: Vec<Arc<AudioGuardrail>>,
        authorization: Arc<crate::providers::auth::NativeAuthorizationServices>,
    ) -> Self {
        Self {
            dispatcher: CustomLoggerRunner::new(callbacks),
            guardrails: CustomGuardrailRunner::new(guardrails),
            authorization,
        }
    }
}

impl Clock for DefaultAudioServices {
    fn now(&self) -> f64 {
        crate::lifecycle::SystemClock.now()
    }
}

impl TerminalDispatcher for DefaultAudioServices {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        self.dispatcher.dispatch(terminal)
    }
}

impl crate::providers::auth::AuthorizationServices for DefaultAudioServices {
    fn environment(&self, key: &str) -> Option<String> {
        self.authorization.environment(key)
    }

    fn signing_time(&self) -> SystemTime {
        self.authorization.signing_time()
    }

    #[cfg(feature = "bedrock-auth")]
    fn resolve_aws_credentials<'a>(
        &'a self,
        config: crate::providers::bedrock::aws_base::AwsAuthConfig,
    ) -> crate::providers::bedrock::aws_base::AwsCredentialFuture<'a> {
        self.authorization.resolve_aws_credentials(config)
    }
}

impl AudioServices for DefaultAudioServices {
    fn guardrails(&self) -> AudioGuardrailRunner {
        self.guardrails.clone()
    }
}

pub struct AudioRoute;

impl AudioRoute {
    pub async fn execute<S: AudioServices>(
        services: &S,
        request: AudioRouteRequest<'_>,
    ) -> ExecutedCall<Value, Error> {
        let provider = resolve_audio_route_provider(request.model, request.custom_llm_provider);
        let context = CallLifecycleContext::new(
            "audio_transcription",
            provider.model,
            provider.custom_llm_provider,
            request
                .litellm_call_id
                .map(str::to_string)
                .unwrap_or_else(new_audio_transcription_call_id),
        )
        .with_metadata(logging_metadata(&request.request_metadata));
        let policy = AudioRequestPolicy {
            guardrail_runner: services.guardrails(),
            request_metadata: request.request_metadata,
        };
        let prepared = PreparedAudioTranscriptionRequest {
            model: provider.model.to_string(),
            custom_llm_provider: provider.custom_llm_provider.to_string(),
            audio: request.audio,
            api_key: request.api_key.map(str::to_string),
            api_base: request.api_base.map(str::to_string),
            extra_headers: request.extra_headers,
            optional_params: request.optional_params,
            timeout: request.timeout,
        };
        CallLifecycle::default()
            .run_prepared(
                context,
                prepared,
                &policy,
                services,
                services,
                |request| std::future::ready(policy.prepare_provider_request(request)),
                |request| execute_audio_transcription_provider_call(services, request),
            )
            .await
    }
}

struct PreparedAudioTranscriptionRequest {
    model: String,
    custom_llm_provider: String,
    audio: AudioInput,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    optional_params: Map<String, Value>,
    timeout: Option<std::time::Duration>,
}

struct AudioRequestPolicy {
    guardrail_runner: AudioGuardrailRunner,
    request_metadata: RequestMetadata,
}

type AudioFuture<'a, T> = Pin<Box<dyn Future<Output = ActionResult<T, Error>> + Send + 'a>>;

impl AudioRequestPolicy {
    async fn run_pre_call_guardrails(
        &self,
        request: PreparedAudioTranscriptionRequest,
    ) -> Result<PreparedAudioTranscriptionRequest, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }
        let guardrail_request = AudioPreCallGuardrailRequest::new(
            request.model.clone(),
            request.custom_llm_provider.clone(),
            request.audio.clone(),
            request.optional_params.clone(),
        );
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_pre_call(
                &guardrail_context(&self.request_metadata),
                guardrail_request,
            )
            .await
            .map_err(guardrail_error_to_core_error)?;
        let (audio, optional_params) = guardrail_request.into_payload();
        Ok(PreparedAudioTranscriptionRequest {
            audio,
            optional_params,
            ..request
        })
    }

    fn prepare_provider_request(
        &self,
        request: PreparedAudioTranscriptionRequest,
    ) -> Result<ProviderAudioTranscriptionRequest, Error> {
        prepare_audio_transcription_provider_call(AudioTranscriptionRequest {
            model: &request.model,
            audio: request.audio,
            api_key: request.api_key.as_deref(),
            api_base: request.api_base.as_deref(),
            custom_llm_provider: Some(&request.custom_llm_provider),
            extra_headers: request.extra_headers,
            optional_params: request.optional_params,
            timeout: request.timeout,
        })
    }

    async fn run_during_call_guardrails(
        &self,
        request: ProviderAudioTranscriptionRequest,
    ) -> Result<ProviderAudioTranscriptionRequest, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }
        let guardrail_request = AudioDuringCallGuardrailRequest::new(
            request.model.clone(),
            request.custom_llm_provider.clone(),
            request.url.clone(),
            request.body.clone(),
        );
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_during_call(
                &guardrail_context(&self.request_metadata),
                guardrail_request,
            )
            .await
            .map_err(guardrail_error_to_core_error)?;
        Ok(ProviderAudioTranscriptionRequest {
            body: guardrail_request.into_body(),
            ..request
        })
    }
}

impl PreCallHooks<PreparedAudioTranscriptionRequest> for AudioRequestPolicy {
    type PreCallFuture<'a>
        = AudioFuture<'a, PreparedAudioTranscriptionRequest>
    where
        Self: 'a;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: PreparedAudioTranscriptionRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move {
            match self.run_pre_call_guardrails(request).await {
                Ok(request) => ActionResult::Replace(request),
                Err(error) => ActionResult::Reject(error),
            }
        })
    }
}

impl ModerationHooks<ProviderAudioTranscriptionRequest> for AudioRequestPolicy {
    type ModerationFuture<'a>
        = AudioFuture<'a, ProviderAudioTranscriptionRequest>
    where
        Self: 'a;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: ProviderAudioTranscriptionRequest,
    ) -> Self::ModerationFuture<'a> {
        Box::pin(async move {
            match self.run_during_call_guardrails(request).await {
                Ok(request) => ActionResult::Replace(request),
                Err(error) => ActionResult::Reject(error),
            }
        })
    }
}

impl DeploymentPreHooks<PreparedAudioTranscriptionRequest> for AudioRequestPolicy {}
impl DeploymentSuccessHooks<Value> for AudioRequestPolicy {}
impl DeploymentFailureHooks for AudioRequestPolicy {}

fn logging_metadata(metadata: &RequestMetadata) -> StandardLoggingMetadata {
    StandardLoggingMetadata {
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

fn new_audio_transcription_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("audio-transcription-{timestamp}-{sequence}")
}
