//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! routes a public model name through `litellm-router-core` (pick, retries, fallbacks, cooldowns)
//! and runs the core route on each attempt.

mod audio_transcription;
mod caching;
mod chat_completions;
mod error;
pub mod messages;
mod ocr;
mod request;
mod responses;
pub mod routing;

use std::{collections::HashMap, sync::Arc};

use arc_swap::ArcSwap;
use axum::{Router, routing::post};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_inference::resources::CoreResources;
use litellm_inference_chat::ChatCompletionsRoute;
use litellm_inference_messages::{MessagesRoute, MessagesShaping};
use litellm_inference_ocr::OcrRoute;
use litellm_inference_responses::ResponsesRoute;
use litellm_inference_transcription::AudioTranscriptionRoute;
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_model_catalog::CostMap;
use litellm_redis_store::RouterStores;
use litellm_secrets::source::SecretSource;

pub use error::{BodyError, Error};
pub use request::{JsonObject, RequestId};
pub use routing::Host;

/// What the gateway knows about models: the installed cost map and what it derives from it. A
/// refresh publishes a new one whole, and an attempt reads one snapshot.
pub struct ModelTable {
    pub cost_map: CostMap,
    /// How the messages route shapes a call to each deployment, by deployment id.
    pub shaping: HashMap<String, MessagesShaping>,
}

pub struct Gateway<H> {
    host: H,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    /// Cooldowns, allowed-fails counters and per-minute usage of the deployments, shared through
    /// the router's Redis once one is attached.
    routing: Arc<RouterStores>,
    models: ArcSwap<ModelTable>,
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

    /// Cooldowns, allowed-fails counters and per-minute usage of the deployments, which the host
    /// hands router-core.
    pub fn routing(&self) -> &RouterStores {
        &self.routing
    }

    /// The model table in force now.
    pub fn models(&self) -> Arc<ModelTable> {
        self.models.load_full()
    }

    /// Replace the model table every later read sees.
    pub fn publish(&self, models: ModelTable) {
        self.models.store(Arc::new(models));
    }

    pub fn new(
        resources: CoreResources,
        http: HttpClientConfig,
        secrets: Arc<dyn SecretSource>,
        router: litellm_router_core::Router,
        models: ModelTable,
        routing: Arc<RouterStores>,
        host: H,
    ) -> Result<Self, litellm_http::Error> {
        let provider = resources.pool.client(&http, ClientVariant::Provider)?;
        let auth = resources.auth.clone();
        Ok(Self {
            host,
            cache: None,
            routing,
            models: ArcSwap::from_pointee(models),
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
