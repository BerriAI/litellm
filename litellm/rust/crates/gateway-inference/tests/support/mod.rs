// Shared across integration-test targets; each target uses a different subset.
#![allow(dead_code)]

use std::sync::Arc;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
    response::Response,
};
use futures_util::future::BoxFuture;
use litellm_callbacks::ErasedCallback;
use litellm_gateway::{
    Callbacks, Gateway,
    host::{NativeHost, NativeRows},
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::Caller;
use litellm_gateway_inference::{RouterTiers, RoutingCache, router};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use litellm_inference::resources::CoreResources;
use litellm_inference_messages::MessagesSettings;
use litellm_model_catalog::{CostMap, IntegrityLimits};
use litellm_secret_value::SecretValue;
use litellm_secrets::source::SecretSource;

use litellm_router_core::{DeploymentSpec, RouterConfig};
use litellm_router_types::RouterSettings;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

pub struct NoSecrets;

impl SecretSource for NoSecrets {
    fn get_secret_str<'a>(
        &'a self,
        _: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async { Ok(None) })
    }
}

pub fn app(model: &str, api_base: &str) -> Router {
    configured_app(model, api_base, None)
}

pub fn app_with_cache(
    model: &str,
    api_base: &str,
    cache: Arc<dyn litellm_cache_response::ResponseCacheService>,
) -> Router {
    configured_app(model, api_base, Some(cache))
}

pub fn app_with_litellm_params(model: &str, api_base: &str, litellm_params: Value) -> Router {
    app_and_gateway(model, api_base, litellm_params).0
}

/// The app with the gateway it serves from, for a test that installs a cost map.
pub fn app_and_gateway(
    model: &str,
    api_base: &str,
    litellm_params: Value,
) -> (Router, Arc<Gateway>) {
    app_with_settings(model, api_base, litellm_params, json!({}))
}

/// The app with the gateway it serves from, under the `litellm_settings` read as `settings`.
pub fn app_with_settings(
    model: &str,
    api_base: &str,
    litellm_params: Value,
    settings: Value,
) -> (Router, Arc<Gateway>) {
    let gateway = Arc::new(hosted_gateway(
        public_model(
            test_params(model, api_base)
                .into_iter()
                .chain(litellm_params.as_object().cloned().unwrap_or_default())
                .collect(),
        ),
        serde_json::from_value(settings).unwrap(),
        None,
        NativeHost::default(),
        &RouterSettings::default(),
    ));
    (mounted(gateway.clone()), gateway)
}

/// The cost map whose document is `rows`.
pub fn cost_map(rows: Value) -> CostMap {
    CostMap::read(
        rows.to_string().as_bytes(),
        None,
        IntegrityLimits::UNCHECKED,
    )
    .unwrap()
}

/// The app whose router is built from `router_settings`.
pub fn app_with_router_settings(
    model: &str,
    api_base: &str,
    litellm_params: Value,
    router_settings: Value,
) -> Router {
    mounted(Arc::new(hosted_gateway(
        public_model(
            test_params(model, api_base)
                .into_iter()
                .chain(litellm_params.as_object().cloned().unwrap_or_default())
                .collect(),
        ),
        MessagesSettings::default(),
        None,
        NativeHost::default(),
        &serde_json::from_value(router_settings).unwrap(),
    )))
}

/// The app with `callbacks` run around each call, in the order given.
pub fn app_with_callbacks(
    model: &str,
    api_base: &str,
    callbacks: Vec<Arc<dyn ErasedCallback>>,
) -> Router {
    app_over_with_callbacks(vec![test_params(model, api_base)], json!({}), callbacks)
}

/// The app whose `public/model` group is one deployment for each of `litellm_params`, with its
/// router built from `router_settings` and `callbacks` run around each call.
pub fn app_over_with_callbacks(
    litellm_params: Vec<Map<String, Value>>,
    router_settings: Value,
    callbacks: Vec<Arc<dyn ErasedCallback>>,
) -> Router {
    mounted(Arc::new(hosted_gateway(
        litellm_params.into_iter().flat_map(public_model).collect(),
        MessagesSettings::default(),
        None,
        NativeHost::new(
            Arc::new(Callbacks::new(callbacks.into_iter().map(Into::into).collect()).unwrap()),
            Arc::default(),
        ),
        &serde_json::from_value(router_settings).unwrap(),
    )))
}

/// The params of a deployment of `model` at `api_base`.
pub fn test_params(model: &str, api_base: &str) -> Map<String, Value> {
    json!({"model": model, "api_base": api_base, "api_key": "test-key", "timeout": 5})
        .as_object()
        .cloned()
        .unwrap()
}

fn configured_app(
    model: &str,
    api_base: &str,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
) -> Router {
    mounted(Arc::new(hosted_gateway(
        public_model(test_params(model, api_base)),
        MessagesSettings::default(),
        cache,
        NativeHost::default(),
        &RouterSettings::default(),
    )))
}

fn mounted(gateway: Arc<Gateway>) -> Router {
    router(gateway).layer(axum::middleware::from_fn(as_the_master_key))
}

fn hosted_gateway(
    model_list: Vec<DeploymentSpec>,
    settings: MessagesSettings,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    host: NativeHost,
    router_settings: &RouterSettings,
) -> Gateway {
    let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let http = Resolution::from(&HttpSettings::default()).config;
    let secrets = Arc::new(NoSecrets);
    let resources = CoreResources::new(pool);
    let (config, dynamic) = RouterConfig::from_settings(router_settings).unwrap();
    let models = litellm_router_core::Router::new(
        RouterConfig {
            model_list,
            ..config
        },
        dynamic,
    )
    .unwrap();
    let gateway = Gateway::new(
        resources,
        http,
        secrets,
        models,
        settings,
        RoutingCache::new(None, RouterTiers::default()),
        host,
    )
    .unwrap();
    match cache {
        Some(cache) => gateway.with_cache(cache),
        None => gateway,
    }
}

/// The one deployment of `public/model`, called with `litellm_params`.
fn public_model(litellm_params: Map<String, Value>) -> Vec<DeploymentSpec> {
    vec![DeploymentSpec {
        model_name: "public/model".into(),
        litellm_params,
        ..DeploymentSpec::default()
    }]
}

/// The app whose router serves `model_list`, each entry as a config spells it, under
/// `router_settings`.
pub fn app_with_models(model_list: Value, router_settings: Value) -> Router {
    mounted(Arc::new(hosted_gateway(
        serde_json::from_value(model_list).unwrap(),
        MessagesSettings::default(),
        None,
        NativeHost::default(),
        &serde_json::from_value(router_settings).unwrap(),
    )))
}

async fn as_the_master_key(
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    request
        .extensions_mut()
        .insert(Authenticated::<NativeRows>::from(Caller::master_key(
            Default::default(),
        )));
    next.run(request).await
}

pub async fn post(app: Router, path: &str, body: Value) -> Response {
    app.oneshot(
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

pub async fn json(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
}

/// Answers with the SSE head and one event, then drops the connection short of the
/// announced body length.
pub async fn truncating_upstream(first_event: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 4096];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{first_event}",
                    first_event.len() * 2
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    base
}
