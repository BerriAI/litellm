//! The proxy's inference endpoints as an axum [`Router`] a server mounts.
//!
//! Authentication, rate limiting and logging are the mounting server's layers; this crate
//! routes a public model name through router-core and runs the core route on each attempt.

mod audio_transcription;
mod caching;
mod chat_completions;
mod error;
pub mod messages;
mod ocr;
mod request;
mod responses;
mod routing;

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
use litellm_router_core::{ConfigError, DeploymentSpec, Router as ModelRouter, RouterConfig};
use litellm_router_types::RouterSettings;
use litellm_secrets::source::SecretSource;

pub use error::{BodyError, Error};
pub use request::{JsonObject, RequestId};

/// The router `settings` describe, serving `model_list` on the gateway's host.
pub fn model_router(
    settings: &RouterSettings,
    model_list: Vec<DeploymentSpec>,
) -> Result<ModelRouter, ConfigError> {
    let (config, dynamic) = RouterConfig::from_settings(settings)?;
    ModelRouter::new(
        RouterConfig {
            model_list,
            ..config
        },
        dynamic,
        Arc::new(routing::GatewayHost::default()),
    )
}

pub struct Gateway {
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    pub audio_transcription: AudioTranscriptionRoute,
    pub chat_completions: ChatCompletionsRoute,
    pub messages: MessagesRoute,
    pub ocr: OcrRoute,
    pub responses: ResponsesRoute,
    pub router: ModelRouter,
    /// How the messages route shapes a call to each deployment, by deployment id.
    shaping: HashMap<String, MessagesShaping>,
    pub secrets: Arc<dyn SecretSource>,
    pub resources: CoreResources,
    pub http: HttpClientConfig,
}

impl Gateway {
    pub(crate) fn shaping(&self, deployment_id: &str) -> MessagesShaping {
        self.shaping.get(deployment_id).cloned().unwrap_or_default()
    }

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
        router: ModelRouter,
        shaping: HashMap<String, MessagesShaping>,
    ) -> Result<Self, litellm_http::Error> {
        let provider = resources.pool.client(&http, ClientVariant::Provider)?;
        let auth = resources.auth.clone();
        Ok(Self {
            cache: None,
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
            shaping,
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

#[cfg(test)]
mod tests {
    use litellm_router_core::{RoutingStrategy, Unsupported};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn built(settings: Value) -> Result<ModelRouter, ConfigError> {
        model_router(&serde_json::from_value(settings).unwrap(), Vec::new())
    }

    #[rstest]
    #[case::pre_call_checks(json!({"enable_pre_call_checks": true}), "enable_pre_call_checks")]
    #[case::tag_filtering(json!({"enable_tag_filtering": true}), "enable_tag_filtering")]
    #[case::named_pre_call_checks(
        json!({"optional_pre_call_checks": ["encrypted_content_affinity"]}),
        "optional_pre_call_checks"
    )]
    #[case::affinity(json!({"model_group_affinity_config": {}}), "model_group_affinity_config")]
    fn a_setting_only_host_selection_serves_fails_boot(
        #[case] settings: Value,
        #[case] expected: &str,
    ) {
        assert!(matches!(
            built(settings),
            Err(ConfigError::Unsupported(Unsupported::Setting(setting))) if setting == expected
        ));
    }

    #[rstest]
    fn a_strategy_only_host_selection_serves_fails_boot() {
        assert!(matches!(
            built(json!({"routing_strategy": "least-busy"})),
            Err(ConfigError::Unsupported(Unsupported::Strategy(
                RoutingStrategy::LeastBusy
            )))
        ));
    }

    #[rstest]
    #[case::unset(json!({}))]
    #[case::checks_off(json!({"enable_pre_call_checks": false, "enable_tag_filtering": false}))]
    #[case::native_strategy(json!({"routing_strategy": "simple-shuffle"}))]
    fn native_routing_settings_boot(#[case] settings: Value) {
        assert!(built(settings).is_ok());
    }
}
