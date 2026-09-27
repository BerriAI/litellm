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
use litellm_core::resources::CoreResources;
use litellm_http::HttpClientConfig;
use litellm_llms::base_llm::ocr::handler::OcrClient;
use litellm_secrets::source::SecretSource;

pub use error::Error;
pub use litellm_router::{Deployment, Router as ModelRouter};
pub use request::{JsonObject, RequestId};

pub struct Gateway {
    pub resources: CoreResources,
    pub http: HttpClientConfig,
    pub secrets: Arc<dyn SecretSource>,
    pub models: ModelRouter,
    pub ocr: OcrClient,
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

fn model_routes() -> Router<Arc<Gateway>> {
    Router::new()
        .route(
            "/chat/completions",
            post(chat_completions::create_from_model_path),
        )
        .route("/embeddings", post(request::unsupported))
        .route("/completions", post(request::unsupported))
}
