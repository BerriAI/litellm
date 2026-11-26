// Shared across integration-test targets; each target uses a different subset.
#![allow(dead_code)]

use std::{collections::HashMap, sync::Arc};

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
    app_with_shaping(
        model,
        api_base,
        litellm_params,
        Value::Object(Default::default()),
    )
}

pub fn app_with_shaping(
    model: &str,
    api_base: &str,
    litellm_params: Value,
    shaping: Value,
) -> Router {
    deployment_app(
        test_params(model, api_base)
            .into_iter()
            .chain(litellm_params.as_object().cloned().unwrap_or_default())
            .collect(),
        shaping,
        None,
    )
}

/// The app whose router is built from `router_settings`.
pub fn app_with_router_settings(
    model: &str,
    api_base: &str,
    litellm_params: Value,
    router_settings: Value,
) -> Router {
    hosted_app(
        test_params(model, api_base)
            .into_iter()
            .chain(litellm_params.as_object().cloned().unwrap_or_default())
            .collect(),
        Value::Object(Map::new()),
        None,
        NativeHost::default(),
        &serde_json::from_value(router_settings).unwrap(),
    )
}

/// The app with `callbacks` run around each call, in the order given.
pub fn app_with_callbacks(
    model: &str,
    api_base: &str,
    callbacks: Vec<Arc<dyn ErasedCallback>>,
) -> Router {
    hosted_app(
        test_params(model, api_base),
        Value::Object(Map::new()),
        None,
        NativeHost::new(
            Arc::new(Callbacks::new(callbacks.into_iter().map(Into::into).collect()).unwrap()),
            Arc::default(),
        ),
        &RouterSettings::default(),
    )
}

fn test_params(model: &str, api_base: &str) -> Map<String, Value> {
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
    deployment_app(
        test_params(model, api_base),
        Value::Object(Map::new()),
        cache,
    )
}

fn deployment_app(
    litellm_params: Map<String, Value>,
    shaping: Value,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
) -> Router {
    hosted_app(
        litellm_params,
        shaping,
        cache,
        NativeHost::default(),
        &RouterSettings::default(),
    )
}

fn hosted_app(
    litellm_params: Map<String, Value>,
    shaping: Value,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    host: NativeHost,
    router_settings: &RouterSettings,
) -> Router {
    let model_list = vec![DeploymentSpec {
        model_name: "public/model".into(),
        litellm_params,
        ..DeploymentSpec::default()
    }];
    let (config, dynamic) = RouterConfig::from_settings(router_settings).unwrap();
    let models = litellm_router_core::Router::new(
        RouterConfig {
            model_list,
            ..config
        },
        dynamic,
    )
    .unwrap();
    let id = models.registry().group("public/model").unwrap()[0]
        .id
        .clone();
    let shaping = HashMap::from([(id, serde_json::from_value(shaping).unwrap())]);
    served_app(models, shaping, cache, host)
}

/// The app whose router serves `model_list`, each entry as a config spells it, under
/// `router_settings`.
pub fn app_with_models(model_list: Value, router_settings: Value) -> Router {
    let (config, dynamic) =
        RouterConfig::from_settings(&serde_json::from_value(router_settings).unwrap()).unwrap();
    let model_list = serde_json::from_value(model_list).unwrap();
    let models = litellm_router_core::Router::new(
        RouterConfig {
            model_list,
            ..config
        },
        dynamic,
    )
    .unwrap();
    served_app(models, HashMap::new(), None, NativeHost::default())
}

fn served_app(
    models: litellm_router_core::Router,
    shaping: HashMap<String, litellm_inference_messages::MessagesShaping>,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    host: NativeHost,
) -> Router {
    let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let http = Resolution::from(&HttpSettings::default()).config;
    let secrets = Arc::new(NoSecrets);
    let resources = CoreResources::new(pool);
    let gateway = Gateway::new(
        resources,
        http,
        secrets,
        models,
        shaping,
        RoutingCache::new(None, RouterTiers::default()),
        host,
    )
    .unwrap();
    let gateway = match cache {
        Some(cache) => gateway.with_cache(cache),
        None => gateway,
    };
    router(Arc::new(gateway)).layer(axum::middleware::from_fn(as_the_master_key))
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
