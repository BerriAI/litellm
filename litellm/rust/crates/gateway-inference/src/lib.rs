//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication is the mounting server's layer; this crate routes a public model name through
//! `litellm-router-core` (pick, retries, fallbacks, cooldowns), runs the core route on each
//! attempt, and tells the host's callback hooks what happened to the call (`hooks.rs`).

mod asked;
mod caching;
mod checked;
mod error;
mod group_usage;
mod headers;
mod hooks;
pub mod messages;
mod model_restamp;
mod ocr;
mod rate_limited;
pub mod record;
mod request;
pub mod routing;
mod routing_cache;
mod serve;
mod stream_event;
mod stream_hold;
mod stream_outcome;
#[cfg(test)]
mod test_support;
mod worker_outcomes;

use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::{Router, routing::post};
use litellm_http::{ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_inference::resources::CoreResources;
use litellm_inference_messages::{MessagesRoute, MessagesSettings};
use litellm_inference_ocr::OcrRoute;
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_model_catalog::CostMap;
use litellm_secrets::source::SecretSource;

pub use asked::call_type;
pub use checked::{BodyCheck, Checked};
pub use error::{BodyError, Ended, Error};
pub use group_usage::{GroupLimits, group_limits};
pub use headers::ExposedHeader;
pub use hooks::{CallHooks, CallType, HookError, Refusal, ServedCall};
pub use litellm_redis_store::RouterTiers;
pub use rate_limited::RateLimited;
pub use request::{JsonObject, MAX_BODY_BYTES, RequestId};
pub use routing::Host;
pub use routing_cache::RoutingCache;

pub struct Gateway<H> {
    host: H,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    routing: Arc<RoutingCache>,
    /// The cost map every request reads. It starts as the bundled one.
    cost_map: ArcSwap<CostMap>,
    /// What `litellm_settings` sets for a messages call to any deployment.
    messages_settings: MessagesSettings,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
    pub router: litellm_router_core::Router,
    pub secrets: Arc<dyn SecretSource>,
    pub resources: CoreResources,
    pub http: HttpClientConfig,
    /// What the request body check reads besides the request.
    body_check: BodyCheck,
}

impl<H: Host> Gateway<H> {
    pub fn with_body_check(self, body_check: BodyCheck) -> Self {
        Self { body_check, ..self }
    }

    pub fn with_cache(self, cache: Arc<dyn litellm_cache_response::ResponseCacheService>) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    /// The deployments' routing state, which the host hands router-core.
    pub fn routing(&self) -> &RoutingCache {
        &self.routing
    }

    /// The routing state itself, so a gateway built for a new config keeps the cooldowns, usage and
    /// health this one counted.
    pub fn shared_routing(&self) -> Arc<RoutingCache> {
        self.routing.clone()
    }

    pub fn host(&self) -> &H {
        &self.host
    }

    /// The cost map a request reads, which stays the same one for as long as the request holds it.
    pub fn cost_map(&self) -> Arc<CostMap> {
        self.cost_map.load_full()
    }

    /// Has every request from now on read `cost_map`.
    pub fn install(&self, cost_map: CostMap) {
        self.cost_map.store(Arc::new(cost_map));
    }

    pub fn new(
        resources: CoreResources,
        http: HttpClientConfig,
        secrets: Arc<dyn SecretSource>,
        router: litellm_router_core::Router,
        messages_settings: MessagesSettings,
        routing: impl Into<Arc<RoutingCache>>,
        host: H,
    ) -> Result<Self, litellm_http::Error> {
        let provider = resources.pool.client(&http, ClientVariant::Provider)?;
        let auth = resources.auth.clone();
        Ok(Self {
            host,
            cache: None,
            routing: routing.into(),
            cost_map: ArcSwap::from_pointee(CostMap::bundled()),
            messages_settings,
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
            body_check: BodyCheck::new(&litellm_config::GeneralSettings::default()),
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
        .layer(axum::middleware::from_fn_with_state(
            gateway.clone(),
            group_usage::told::<H>,
        ))
        .layer(axum::middleware::from_fn(headers::render_response_headers))
        .layer(axum::middleware::from_fn(record::scoped))
        .with_state(gateway)
}
