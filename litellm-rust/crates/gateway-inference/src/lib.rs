//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! maps a public model name to its deployment and runs the core route.

mod audio_transcription;
mod chat_completions;
mod error;
pub mod messages;
mod ocr;
mod request;

use std::sync::Arc;

use axum::{Router, routing::post};
use litellm_core::{
    audio_transcription::AudioTranscriptionRoute, chat_completions::ChatCompletionsRoute,
    messages::MessagesRoute, ocr::OcrRoute, resources::CoreResources,
};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_secrets::source::SecretSource;

pub use error::Error;
pub use litellm_router::{Deployment, Router as ModelList};

pub struct Gateway {
    pub audio_transcription: AudioTranscriptionRoute,
    pub chat_completions: ChatCompletionsRoute,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
    pub models: ModelList,
    pub secrets: Arc<dyn SecretSource>,
    pub resources: CoreResources,
    pub http: HttpClientConfig,
}

impl Gateway {
    pub fn new(
        resources: CoreResources,
        http: HttpClientConfig,
        secrets: Arc<dyn SecretSource>,
        models: ModelList,
    ) -> Result<Self, litellm_http::Error> {
        let provider = resources.pool.client(&http, ClientVariant::Provider)?;
        let auth = resources.auth.clone();
        Ok(Self {
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
            messages: MessagesRoute::new(provider, auth.clone(), secrets.clone()),
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
        .route("/engines/{*path}", post(chat_completions::deployment))
        .route(
            "/openai/deployments/{*path}",
            post(chat_completions::deployment),
        )
        .route("/audio/transcriptions", post(audio_transcription::create))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcription::create),
        )
        .route("/responses", post(request::unsupported))
        .route("/v1/responses", post(request::unsupported))
        .route("/embeddings", post(request::unsupported))
        .route("/v1/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
        .route("/v1/completions", post(request::unsupported))
        .layer(axum::extract::DefaultBodyLimit::max(
            request::MAX_BODY_BYTES,
        ))
        .with_state(gateway)
}
