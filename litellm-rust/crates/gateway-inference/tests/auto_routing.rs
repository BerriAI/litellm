mod support;

use axum::{
    Json,
    response::{IntoResponse, Response},
};
use litellm_gateway_auth::Permissions;
use litellm_gateway_inference::{
    Deployment, Error, Gateway,
    auto_routing::{ApiFormat, AutoRoutingStrategy, StrategyRequest},
};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use litellm_inference::resources::CoreResources;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct EchoStrategy(Arc<AtomicUsize>);

impl AutoRoutingStrategy for EchoStrategy {
    fn execute<'a>(
        &'a self,
        _: &'a Gateway,
        request: StrategyRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Response, Error>> + Send + 'a>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if request.body.get("fail").and_then(Value::as_bool) == Some(true) {
                return Err(Error::InvalidBody("strategy rejected request".into()));
            }
            let format = match request.format {
                ApiFormat::Chat => "chat",
                ApiFormat::Responses => "responses",
                ApiFormat::Messages => "messages",
            };
            Ok(Json(json!({"format": format, "body": request.body, "request_id": request.headers.get("x-request-id").and_then(|v| v.to_str().ok()), "no_store": request.cache_options.policy.no_store})).into_response())
        })
    }
}

#[fixture]
fn gateway() -> Gateway {
    Gateway::new(
        CoreResources::new(Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)))),
        Resolution::from(&HttpSettings::default()).config,
        Arc::new(support::NoSecrets),
        [(
            "virtual".into(),
            Deployment {
                model: "virtual-target".into(),
                ..Default::default()
            },
        )]
        .into_iter()
        .collect(),
    )
    .unwrap()
}

#[rstest]
#[case::chat("/v1/chat/completions", "chat", json!({"model":"virtual"}))]
#[case::chat_alias("/chat/completions", "chat", json!({"model":"virtual"}))]
#[case::responses("/v1/responses", "responses", json!({"model":"virtual"}))]
#[case::responses_alias("/responses", "responses", json!({"model":"virtual"}))]
#[case::messages("/v1/messages", "messages", json!({"model":"virtual"}))]
#[case::engine("/engines/virtual/chat/completions", "chat", json!({}))]
#[case::deployment("/openai/deployments/virtual/chat/completions", "chat", json!({"model":null}))]
#[case::body_wins("/engines/unknown/chat/completions", "chat", json!({"model":"virtual"}))]
#[tokio::test]
async fn virtual_models_invoke_the_strategy_once(
    gateway: Gateway,
    #[case] path: &str,
    #[case] format: &str,
    #[case] body: Value,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let gateway = gateway
        .with_auto_routing_strategy("virtual".into(), Arc::new(EchoStrategy(calls.clone())))
        .unwrap();
    let response = support::post(
        support::authenticated_app(gateway, Permissions::All, None),
        path,
        body,
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["format"], format);
    assert_eq!(body["body"]["model"], "virtual");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::chat("/v1/chat/completions")]
#[case::responses("/v1/responses")]
#[case::messages("/v1/messages")]
#[tokio::test]
async fn denied_virtual_models_never_invoke_the_strategy(gateway: Gateway, #[case] path: &str) {
    let calls = Arc::new(AtomicUsize::new(0));
    let gateway = gateway
        .with_auto_routing_strategy("virtual".into(), Arc::new(EchoStrategy(calls.clone())))
        .unwrap();
    let response = support::post(
        support::authenticated_app(gateway, Permissions::Only(Arc::from([])), None),
        path,
        json!({"model":"virtual"}),
    )
    .await;
    assert_eq!(response.status(), 403);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[rstest]
#[case::chat("/v1/chat/completions", false)]
#[case::responses("/v1/responses", false)]
#[case::messages("/v1/messages", true)]
#[tokio::test]
async fn strategy_errors_keep_the_callers_error_format(
    gateway: Gateway,
    #[case] path: &str,
    #[case] anthropic: bool,
) {
    let gateway = gateway
        .with_auto_routing_strategy(
            "virtual".into(),
            Arc::new(EchoStrategy(Arc::new(AtomicUsize::new(0)))),
        )
        .unwrap();
    let response = support::post(
        support::authenticated_app(gateway, Permissions::All, None),
        path,
        json!({"model":"virtual", "fail":true}),
    )
    .await;
    assert_eq!(response.status(), 400);
    let body = support::json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("strategy rejected request")
    );
    assert_eq!(body.get("type").is_some(), anthropic);
}

#[rstest]
#[case::chat("/v1/chat/completions")]
#[case::responses("/v1/responses")]
#[case::messages("/v1/messages")]
#[tokio::test]
async fn cache_controls_are_prepared_before_strategy_execution(
    gateway: Gateway,
    #[case] path: &str,
) {
    let gateway = gateway
        .with_auto_routing_strategy(
            "virtual".into(),
            Arc::new(EchoStrategy(Arc::new(AtomicUsize::new(0)))),
        )
        .unwrap();
    let response = support::post(
        support::authenticated_app(gateway, Permissions::All, None),
        path,
        json!({"model":"virtual", "cache":{"no-store":true}, "caching":false}),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = support::json(response).await;
    assert_eq!(body["no_store"], true);
    assert!(body["body"].get("cache").is_none());
    assert!(body["body"].get("caching").is_none());
}

#[rstest]
#[case::chat("/v1/chat/completions")]
#[case::responses("/v1/responses")]
#[case::messages("/v1/messages")]
#[tokio::test]
async fn incoming_headers_reach_the_strategy(gateway: Gateway, #[case] path: &str) {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let gateway = gateway
        .with_auto_routing_strategy(
            "virtual".into(),
            Arc::new(EchoStrategy(Arc::new(AtomicUsize::new(0)))),
        )
        .unwrap();
    let response = support::authenticated_app(gateway, Permissions::All, None)
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("x-request-id", "request-context")
                .body(Body::from(r#"{"model":"virtual"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        support::json(response).await["request_id"],
        "request-context"
    );
}

#[rstest]
#[case::chat("/v1/chat/completions", false)]
#[case::responses("/v1/responses", false)]
#[case::messages("/v1/messages", true)]
#[tokio::test]
async fn malformed_bodies_preserve_api_error_envelopes(
    gateway: Gateway,
    #[case] path: &str,
    #[case] anthropic: bool,
) {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    let calls = Arc::new(AtomicUsize::new(0));
    let gateway = gateway
        .with_auto_routing_strategy("virtual".into(), Arc::new(EchoStrategy(calls.clone())))
        .unwrap();
    let response = support::authenticated_app(gateway, Permissions::All, None)
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("x-request-id", "malformed-request")
                .body(Body::from("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body = support::json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body.get("type").is_some(), anthropic);
    if anthropic {
        assert_eq!(body["request_id"], "malformed-request");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[rstest]
#[case::chat("/v1/chat/completions")]
#[case::responses("/v1/responses")]
#[case::messages("/v1/messages")]
#[tokio::test]
async fn invalid_cache_settings_do_not_invoke_the_strategy(gateway: Gateway, #[case] path: &str) {
    let calls = Arc::new(AtomicUsize::new(0));
    let gateway = gateway
        .with_auto_routing_strategy("virtual".into(), Arc::new(EchoStrategy(calls.clone())))
        .unwrap();
    let response = support::post(
        support::authenticated_app(gateway, Permissions::All, None),
        path,
        json!({"model":"virtual", "cache":{"ttl":-1}}),
    )
    .await;
    assert_eq!(response.status(), 400);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
