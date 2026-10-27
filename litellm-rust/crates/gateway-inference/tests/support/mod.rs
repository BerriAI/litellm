// Shared across integration-test targets; each target uses a different subset.
#![allow(dead_code)]

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
    response::Response,
};
use futures_util::future::BoxFuture;
use litellm_gateway_inference::{Deployment, Gateway, router};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use litellm_inference::resources::CoreResources;
use litellm_secrets::{SecretValue, source::SecretSource};
use serde_json::Value;
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

fn configured_app(
    model: &str,
    api_base: &str,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
) -> Router {
    let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let http = Resolution::from(&HttpSettings::default()).config;
    let secrets = Arc::new(NoSecrets);
    let resources = CoreResources::new(pool);
    let gateway = Gateway::new(
        resources,
        http,
        secrets,
        [(
            "public/model".into(),
            Deployment {
                model: model.into(),
                api_base: Some(api_base.into()),
                api_key: Some("test-key".into()),
                timeout: Some(Duration::from_secs(5)),
                ..Default::default()
            },
        )]
        .into_iter()
        .collect(),
    )
    .unwrap();
    let gateway = match cache {
        Some(cache) => gateway.with_cache(cache),
        None => gateway,
    };
    router(Arc::new(gateway)).layer(axum::middleware::from_fn(master_caller))
}

async fn master_caller(
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    request
        .extensions_mut()
        .insert(litellm_gateway_auth::Caller::Master);
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
