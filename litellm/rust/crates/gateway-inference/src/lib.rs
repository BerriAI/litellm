//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! routes a public model name through `litellm-router-core` (pick, retries, fallbacks, cooldowns)
//! and runs the core route on each attempt.

mod audio_transcription;
mod caching;
mod chat_completions;
mod error;
mod headers;
pub mod messages;
mod ocr;
mod request;
mod responses;
pub mod routing;
mod stream_outcome;

use std::{collections::HashMap, sync::Arc};

use axum::{Router, routing::post};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_inference::resources::CoreResources;
use litellm_inference_chat::ChatCompletionsRoute;
use litellm_inference_messages::{MessagesRoute, MessagesShaping};
use litellm_inference_ocr::OcrRoute;
use litellm_inference_responses::ResponsesRoute;
use litellm_inference_transcription::AudioTranscriptionRoute;
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_secrets::source::SecretSource;

pub use error::{BodyError, Ended, Error};
pub use headers::ExposedHeader;
pub use request::{JsonObject, RequestId};
pub use routing::Host;

pub struct Gateway<H> {
    host: H,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    client_side_credentials: bool,
    /// Cooldowns, allowed-fails counters and per-minute usage of the deployments, in this process.
    routing: Arc<litellm_router_core::InMemoryRouterCache>,
    /// How the chat and messages routes shape a call to each deployment, by deployment id.
    shaping: HashMap<String, MessagesShaping>,
    pub audio_transcription: AudioTranscriptionRoute,
    pub chat_completions: ChatCompletionsRoute,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
    pub responses: ResponsesRoute,
    pub router: litellm_router_core::Router,
    pub secrets: Arc<dyn SecretSource>,
    pub resources: CoreResources,
    pub http: HttpClientConfig,
}

impl<H: Host> Gateway<H> {
    pub fn with_cache(self, cache: Arc<dyn litellm_cache_response::ResponseCacheService>) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    pub fn with_client_side_credentials(self, allowed: bool) -> Self {
        Self {
            client_side_credentials: allowed,
            ..self
        }
    }

    /// Cooldowns, allowed-fails counters and per-minute usage of the deployments, which the host
    /// hands router-core.
    pub fn routing(&self) -> &litellm_router_core::InMemoryRouterCache {
        &self.routing
    }

    /// How the chat and messages routes shape a call to deployment `id`.
    pub(crate) fn shaping(&self, id: &str) -> MessagesShaping {
        self.shaping.get(id).cloned().unwrap_or_default()
    }

    pub fn new(
        resources: CoreResources,
        http: HttpClientConfig,
        secrets: Arc<dyn SecretSource>,
        router: litellm_router_core::Router,
        shaping: HashMap<String, MessagesShaping>,
        host: H,
    ) -> Result<Self, litellm_http::Error> {
        let provider = resources.pool.client(&http, ClientVariant::Provider)?;
        let auth = resources.auth.clone();
        Ok(Self {
            host,
            cache: None,
            client_side_credentials: false,
            routing: Arc::default(),
            shaping,
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
            router,
            secrets,
            resources,
            http,
        })
    }
}

pub fn router<H: Host>(gateway: Arc<Gateway<H>>) -> Router {
    Router::new()
        .route("/v1/messages", post(messages::create::<H>))
        .route("/ocr", post(ocr::create::<H>))
        .route("/v1/ocr", post(ocr::create::<H>))
        .route("/chat/completions", post(chat_completions::create::<H>))
        .route("/v1/chat/completions", post(chat_completions::create::<H>))
        .nest("/engines/{model}", model_routes::<H>())
        .nest("/openai/deployments/{model}", model_routes::<H>())
        .route(
            "/audio/transcriptions",
            post(audio_transcription::create::<H>),
        )
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcription::create::<H>),
        )
        .route("/responses", post(responses::create::<H>))
        .route("/v1/responses", post(responses::create::<H>))
        .route("/embeddings", post(request::unsupported))
        .route("/v1/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
        .route("/v1/completions", post(request::unsupported))
        .layer(axum::extract::DefaultBodyLimit::max(
            request::MAX_BODY_BYTES,
        ))
        .layer(headers::layer())
        .with_state(gateway)
}

fn model_routes<H: Host>() -> Router<Arc<Gateway<H>>> {
    Router::new()
        .route(
            "/chat/completions",
            post(chat_completions::create_from_model_path::<H>),
        )
        .route("/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
}
