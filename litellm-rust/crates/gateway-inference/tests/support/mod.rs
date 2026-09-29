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
use litellm_core::resources::CoreResources;
use litellm_gateway_inference::{Deployment, Gateway, router};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
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
    app_with_permissions(model, api_base, litellm_gateway_auth::Permissions::All)
}

pub fn app_with_permissions(
    model: &str,
    api_base: &str,
    permissions: litellm_gateway_auth::Permissions,
) -> Router {
    configured_app(model, api_base, permissions, None, None, None)
}

pub fn app_with_cache(
    model: &str,
    api_base: &str,
    cache: Arc<dyn litellm_cache_response::ResponseCacheService>,
) -> Router {
    configured_app(
        model,
        api_base,
        litellm_gateway_auth::Permissions::All,
        Some(cache),
        None,
        None,
    )
}

pub fn app_with_cache_for_principal(
    model: &str,
    api_base: &str,
    cache: Arc<dyn litellm_cache_response::ResponseCacheService>,
    principal: litellm_gateway_auth::Principal,
) -> Router {
    configured_app(
        model,
        api_base,
        litellm_gateway_auth::Permissions::All,
        Some(cache),
        Some(principal),
        None,
    )
}

pub fn app_with_accounting(
    model: &str,
    api_base: &str,
    accounting: Option<litellm_gateway_inference::accounting::ResponsesAccounting>,
    permissions: litellm_gateway_auth::Permissions,
) -> Router {
    configured_app(model, api_base, permissions, None, None, accounting)
}

fn configured_app(
    model: &str,
    api_base: &str,
    permissions: litellm_gateway_auth::Permissions,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
    principal: Option<litellm_gateway_auth::Principal>,
    accounting: Option<litellm_gateway_inference::accounting::ResponsesAccounting>,
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
    let gateway = match accounting {
        Some(accounting) => gateway.with_responses_accounting(accounting),
        None => gateway,
    };
    router(Arc::new(gateway)).layer(axum::middleware::from_fn_with_state(
        (permissions, principal),
        test_identity,
    ))
}

async fn test_identity(
    axum::extract::State((permissions, principal)): axum::extract::State<(
        litellm_gateway_auth::Permissions,
        Option<litellm_gateway_auth::Principal>,
    )>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let auth = litellm_gateway_auth::Auth::new(
        Arc::new(litellm_gateway_auth::MasterKeyAuthenticator::new(
            Some(SecretValue::new("test-inbound-key")),
            Arc::new(NoSecrets),
        )),
        Arc::new(TestPermissions(permissions, principal)),
        Arc::new(litellm_gateway_auth::NoAdditionalPolicy),
        Arc::new(litellm_gateway_auth::SystemClock),
    );
    let identity = auth
        .authenticate(&SecretValue::new("test-inbound-key"))
        .await
        .unwrap();
    request.extensions_mut().insert(identity);
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

struct TestPermissions(
    litellm_gateway_auth::Permissions,
    Option<litellm_gateway_auth::Principal>,
);

impl litellm_gateway_auth::IdentityResolver for TestPermissions {
    fn resolve<'a>(
        &'a self,
        identity: &'a litellm_gateway_auth::VerifiedIdentity,
    ) -> litellm_gateway_auth::AuthFuture<'a, litellm_gateway_auth::ResolvedIdentity> {
        Box::pin(async move {
            Ok(litellm_gateway_auth::ResolvedIdentity {
                principal: self.1.clone().unwrap_or_else(|| identity.principal.clone()),
                permissions: self.0.clone(),
            })
        })
    }
}
