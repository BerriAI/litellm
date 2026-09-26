use std::sync::Arc;

use axum::{Router, extract::Request};
use tower_http::trace::{DefaultOnResponse, TraceLayer};

use litellm_config::Config;
use litellm_core::resources::CoreResources;
use litellm_gateway_auth::{Auth, RequireMasterKey};
use litellm_gateway_inference::{Gateway, ModelList};
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
};
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use litellm_secrets::source::EnvironmentSecrets;

pub fn build_inference(config: &Config) -> Result<Arc<Gateway>, litellm_http::Error> {
    let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let http = Resolution::from(&HttpSettings::default()).config;
    let client = pool.client(&http, ClientVariant::Provider)?;
    let secrets = Arc::new(EnvironmentSecrets::python_compatible(client));
    let resources = CoreResources::new(pool);
    let ocr = resources.ocr_client(
        &http,
        Default::default(),
        OcrSettings::default(),
        secrets.clone(),
    )?;

    Ok(Arc::new(Gateway {
        resources,
        http,
        secrets,
        models: ModelList::from_model_list(&config.model_list),
        ocr,
    }))
}

pub fn router(inference: Arc<Gateway>, config: &Config) -> Router {
    let auth = Auth::from_config(config, inference.secrets.clone());
    litellm_gateway_inference::router(inference)
        .route_layer(axum::middleware::from_extractor_with_state::<
            RequireMasterKey,
            _,
        >(auth))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request| {
                    tracing::info_span!("request", method = %request.method(), path = request.uri().path())
                })
                .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
        )
}
