//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! routes a public model name through `litellm-router-core` (pick, retries, fallbacks, cooldowns)
//! and runs the core route on each attempt.

mod caching;
mod error;
mod headers;
pub mod messages;
mod ocr;
mod request;
pub mod routing;
mod stream_outcome;

use std::{collections::HashMap, sync::Arc};

use axum::{Router, routing::post};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_inference::resources::CoreResources;
use litellm_inference_messages::{MessagesRoute, MessagesShaping};
use litellm_inference_ocr::OcrRoute;
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_secrets::source::SecretSource;

pub use error::{BodyError, Ended, Error};
pub use headers::ExposedHeader;
pub use request::{JsonObject, RequestId};
pub use routing::Host;

pub struct Gateway<H> {
    host: H,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    /// Cooldowns, allowed-fails counters and per-minute usage of the deployments, in this process.
    routing: Arc<litellm_router_core::InMemoryRouterCache>,
    /// How the messages route shapes a call to each deployment, by deployment id.
    shaping: HashMap<String, MessagesShaping>,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
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
    pub fn routing(&self) -> &litellm_router_core::InMemoryRouterCache {
        &self.routing
    }

    /// How the messages route shapes a call to deployment `id`.
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
            routing: Arc::default(),
            shaping,
            messages: MessagesRoute::new(provider, auth.clone(), secrets.clone()),
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
        .layer(axum::extract::DefaultBodyLimit::max(
            request::MAX_BODY_BYTES,
        ))
        .layer(headers::layer())
        .with_state(gateway)
}
