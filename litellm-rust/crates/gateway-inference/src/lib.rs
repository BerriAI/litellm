//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! maps a public model name to its deployment and runs the core route.

mod audio_transcription;
pub mod auto_routing;
mod caching;
mod chat_completions;
mod error;
pub mod messages;
mod ocr;
mod request;
mod responses;

use std::sync::Arc;

use axum::{Router, routing::post};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_inference::resources::CoreResources;
use litellm_inference_chat::ChatCompletionsRoute;
use litellm_inference_messages::MessagesRoute;
use litellm_inference_ocr::OcrRoute;
use litellm_inference_responses::ResponsesRoute;
use litellm_inference_transcription::AudioTranscriptionRoute;
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_secrets::source::SecretSource;

pub use error::AutoRoutingConfigError;
pub use error::Error;
pub use litellm_router::{Deployment, Router as ModelRouter};
pub use request::{JsonObject, RequestId};

pub struct Gateway {
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    auto_routing: auto_routing::Registry,
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
            auto_routing: auto_routing::Registry::default(),
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

    pub fn with_auto_routing_strategy(
        mut self,
        model: String,
        strategy: Arc<dyn auto_routing::AutoRoutingStrategy>,
    ) -> Result<Self, AutoRoutingConfigError> {
        if self.models.get(&model).is_none() {
            return Err(AutoRoutingConfigError {
                model,
                reason: "model is not configured".into(),
            });
        }
        self.auto_routing.insert(model, strategy);
        Ok(self)
    }

    pub fn with_auto_routing_models(
        mut self,
        models: &[litellm_config::Model],
    ) -> Result<Self, AutoRoutingConfigError> {
        self.auto_routing = auto_routing::Registry::from_models(models, &self.models)?;
        Ok(self)
    }
}

pub fn router(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/messages", post(auto_routing::messages))
        .route("/ocr", post(ocr::create))
        .route("/v1/ocr", post(ocr::create))
        .route("/chat/completions", post(auto_routing::chat))
        .route("/v1/chat/completions", post(auto_routing::chat))
        .nest("/engines/{model}", model_routes())
        .nest("/openai/deployments/{model}", model_routes())
        .route("/audio/transcriptions", post(audio_transcription::create))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcription::create),
        )
        .route("/responses", post(auto_routing::responses))
        .route("/v1/responses", post(auto_routing::responses))
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
            post(auto_routing::chat_from_model_path),
        )
        .route("/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
}
