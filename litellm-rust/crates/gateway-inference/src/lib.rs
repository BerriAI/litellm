//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! maps a public model name to its deployment and runs the core route.

mod audio_transcription;
mod caching;
mod chat_completions;
mod client_side;
mod error;
mod hooks;
pub mod messages;
pub mod observe;
mod ocr;
mod request;
mod responses;

use std::sync::{Arc, OnceLock};

use axum::{Router, routing::post};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_inference::resources::CoreResources;
use litellm_inference_chat::ChatCompletionsRoute;
use litellm_inference_messages::MessagesRoute;
use litellm_inference_ocr::OcrRoute;
use litellm_inference_responses::ResponsesRoute;
use litellm_inference_transcription::AudioTranscriptionRoute;
use litellm_llms::base_llm::{
    model_info::ModelInfo,
    ocr::{handler::OcrClient, settings::OcrSettings},
};
use litellm_secrets::source::SecretSource;

pub use error::Error;
pub use hooks::Hooks;
pub use litellm_router::{Deployment, Error as RoutingError, Router as ModelRouter};
pub use request::{JsonBody, JsonObject, RequestId};

pub struct Gateway {
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    hooks: OnceLock<Hooks>,
    model_info: OnceLock<Arc<dyn ModelInfo>>,
    drop_params: bool,
    allow_client_side_credentials: bool,
    pub audio_transcription: AudioTranscriptionRoute,
    pub chat_completions: ChatCompletionsRoute,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
    pub responses: ResponsesRoute,
    pub models: ModelRouter,
    pub secrets: Arc<dyn SecretSource>,
    pub resources: CoreResources,
    pub http: HttpClientConfig,
}

impl Gateway {
    pub fn with_cache(self, cache: Arc<dyn litellm_cache_response::ResponseCacheService>) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    /// Runs `/v1/messages` and chat completions requests through `hooks`. Only the first call
    /// takes effect.
    pub fn install_hooks(&self, hooks: Hooks) {
        let _ = self.hooks.set(hooks);
    }

    /// The hooks requests run through, once installed.
    pub fn hooks(&self) -> Option<&Hooks> {
        self.hooks.get()
    }

    /// Gives providers whose param rules read model capabilities the model cost map. Only the
    /// first call takes effect.
    pub fn install_model_info(&self, model_info: Arc<dyn ModelInfo>) {
        let _ = self.model_info.set(model_info);
    }

    pub(crate) fn model_info(&self) -> Option<Arc<dyn ModelInfo>> {
        self.model_info.get().cloned()
    }

    /// `litellm_settings.drop_params`: drops params a provider does not support instead of
    /// refusing the request.
    pub fn with_drop_params(self, drop_params: bool) -> Self {
        Self {
            drop_params,
            ..self
        }
    }

    pub(crate) fn drop_params(&self) -> bool {
        self.drop_params
    }

    /// `general_settings.allow_client_side_credentials`: lets a request body name the
    /// connection and credential params the proxy otherwise refuses.
    pub fn with_client_side_credentials(self, allowed: bool) -> Self {
        Self {
            allow_client_side_credentials: allowed,
            ..self
        }
    }

    /// The proxy's `is_request_body_safe`, run on every inference body before routing.
    pub(crate) fn check_body(
        &self,
        body: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error> {
        Ok(client_side::check(
            body,
            self.allow_client_side_credentials,
        )?)
    }

    pub fn new(
        resources: CoreResources,
        http: HttpClientConfig,
        secrets: Arc<dyn SecretSource>,
        models: ModelRouter,
    ) -> Result<Self, litellm_http::Error> {
        let provider = resources.pool.client(&http, ClientVariant::Provider)?;
        let auth = resources.auth.clone();
        Ok(Self {
            cache: None,
            hooks: OnceLock::new(),
            model_info: OnceLock::new(),
            drop_params: false,
            allow_client_side_credentials: false,
            audio_transcription: AudioTranscriptionRoute::new(
                provider.clone(),
                auth.clone(),
                secrets.clone(),
            ),
            chat_completions: ChatCompletionsRoute::new(
                provider.clone(),
                auth.clone(),
                secrets.clone(),
            ),
            messages: MessagesRoute::new(provider.clone(), auth.clone(), secrets.clone()),
            responses: ResponsesRoute::new(provider, auth.clone(), secrets.clone()),
            ocr: OcrRoute::new(OcrClient::new(
                &resources.pool,
                &http,
                UrlPolicy::default(),
                auth,
                OcrSettings::default(),
                secrets.clone(),
            )?),
            models,
            secrets,
            resources,
            http,
        })
    }
}

pub fn router(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/messages", post(messages::create))
        .route("/ocr", post(ocr::create))
        .route("/v1/ocr", post(ocr::create))
        .route("/chat/completions", post(chat_completions::create))
        .route("/v1/chat/completions", post(chat_completions::create))
        .nest("/engines/{model}", model_routes())
        .nest("/openai/deployments/{model}", model_routes())
        .route("/audio/transcriptions", post(audio_transcription::create))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcription::create),
        )
        .route("/responses", post(responses::create))
        .route("/v1/responses", post(responses::create))
        .route("/embeddings", post(request::unsupported))
        .route("/v1/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
        .route("/v1/completions", post(request::unsupported))
        .layer(axum::extract::DefaultBodyLimit::max(
            request::MAX_BODY_BYTES,
        ))
        .with_state(gateway)
}

fn model_routes() -> Router<Arc<Gateway>> {
    Router::new()
        .route(
            "/chat/completions",
            post(chat_completions::create_from_model_path),
        )
        .route("/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
}
