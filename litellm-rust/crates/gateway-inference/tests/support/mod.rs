use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
    response::Response,
};
use futures_util::future::BoxFuture;
use litellm_core::resources::CoreResources;
use litellm_gateway_inference::{Deployment, Gateway, router};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use litellm_llms::base_llm::ocr::settings::OcrSettings;
use litellm_secrets::{SecretValue, source::SecretSource};
use serde_json::Value;
use tower::ServiceExt;

struct NoSecrets;

impl SecretSource for NoSecrets {
    fn get_secret_str<'a>(
        &'a self,
        _: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async { Ok(None) })
    }
}

pub fn app(model: &str, api_base: &str) -> Router {
    let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let http = Resolution::from(&HttpSettings::default()).config;
    let secrets = Arc::new(NoSecrets);
    let resources = CoreResources::new(pool);
    let ocr = resources
        .ocr_client(
            &http,
            Default::default(),
            OcrSettings::default(),
            secrets.clone(),
        )
        .unwrap();
    router(Arc::new(Gateway {
        resources,
        http,
        secrets,
        ocr,
        models: [(
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
    }))
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
