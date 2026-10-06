//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! maps a public model name to its deployment and runs the core route.

mod audio_transcription;
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

pub use error::Error;
use litellm_host::interceptors::Interceptors;
use litellm_inference::RouteError;
pub use litellm_router::{Deployment, Router as ModelRouter, RouterHooks};
pub use request::{JsonObject, RequestId};

pub struct Gateway<R = (), I = ()> {
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    pub audio_transcription: AudioTranscriptionRoute,
    pub chat_completions: ChatCompletionsRoute,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
    pub responses: ResponsesRoute,
    pub models: ModelRouter<R>,
    pub interceptors: I,
    pub secrets: Arc<dyn SecretSource>,
    pub resources: CoreResources,
    pub http: HttpClientConfig,
}

impl Gateway {
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
            interceptors: (),
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

impl<R, I> Gateway<R, I> {
    pub fn with_cache(self, cache: Arc<dyn litellm_cache_response::ResponseCacheService>) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    pub fn with_hooks<T, H>(self, router_hooks: T, interceptors: H) -> Gateway<T, H> {
        Gateway {
            cache: self.cache,
            audio_transcription: self.audio_transcription,
            chat_completions: self.chat_completions,
            messages: self.messages,
            ocr: self.ocr,
            responses: self.responses,
            models: self.models.with_hooks(router_hooks),
            interceptors,
            secrets: self.secrets,
            resources: self.resources,
            http: self.http,
        }
    }
}

pub fn router<R, I>(gateway: Arc<Gateway<R, I>>) -> Router
where
    R: RouterHooks + 'static,
    I: Interceptors<RouteError> + Clone + 'static,
{
    Router::new()
        .route("/v1/messages", post(messages::create::<R, I>))
        .route("/ocr", post(ocr::create::<R, I>))
        .route("/v1/ocr", post(ocr::create::<R, I>))
        .route("/chat/completions", post(chat_completions::create::<R, I>))
        .route(
            "/v1/chat/completions",
            post(chat_completions::create::<R, I>),
        )
        .nest("/engines/{model}", model_routes::<R, I>())
        .nest("/openai/deployments/{model}", model_routes::<R, I>())
        .route(
            "/audio/transcriptions",
            post(audio_transcription::create::<R, I>),
        )
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcription::create::<R, I>),
        )
        .route("/responses", post(responses::create::<R, I>))
        .route("/v1/responses", post(responses::create::<R, I>))
        .route("/embeddings", post(request::unsupported))
        .route("/v1/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
        .route("/v1/completions", post(request::unsupported))
        .layer(axum::extract::DefaultBodyLimit::max(
            request::MAX_BODY_BYTES,
        ))
        .with_state(gateway)
}

fn model_routes<R, I>() -> Router<Arc<Gateway<R, I>>>
where
    R: RouterHooks + 'static,
    I: Interceptors<RouteError> + Clone + 'static,
{
    Router::new()
        .route(
            "/chat/completions",
            post(chat_completions::create_from_model_path::<R, I>),
        )
        .route("/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
}
