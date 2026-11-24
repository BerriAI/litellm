//! The Prometheus logger through the native app: a served call and a failed one are recorded, and
//! `/metrics` writes them for a proxy admin.

use std::{io::Read, sync::Arc, time::Duration};

use axum::{body::Body, http::Request, response::Response};
use flate2::read::GzDecoder;
use litellm_config::{Config, Environment};
use litellm_gateway::{AppState, RedisClients, Resources};
use litellm_license::Entitlements;
use rstest::rstest;
use serde_json::json;
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

const MASTER_KEY: &str = "gateway-key";

fn config(upstream: &MockServer) -> Config {
    Config::from_yaml(&format!(
        "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n    model_info:\n      id: claude-1\nrouter_settings:\n  num_retries: 0\nlitellm_settings:\n  callbacks: [prometheus]\ngeneral_settings:\n  master_key: {MASTER_KEY}\n",
        upstream.uri()
    ))
    .unwrap()
}

async fn app(config: &Config) -> axum::Router {
    let inference = litellm_gateway::build_inference(
        config,
        &Environment::default(),
        &Resources::build(config).unwrap(),
        litellm_gateway_inference::RoutingCache::new(
            None,
            litellm_gateway_inference::RouterTiers::default(),
        ),
        &RedisClients::default(),
    )
    .await
    .unwrap();
    let state = AppState {
        key_limiter: inference.key_limiter,
        inference: inference.gateway,
        metrics: inference.metrics,
        redis: RedisClients::default(),
        database: None,
        keys: None,
        entitlements: Arc::new(Entitlements::Community),
        environment: Arc::new(|_: &str| None::<String>),
        gateway_requests: None,
        gateway_requests_flush: None,
    };
    litellm_gateway::middleware::global(
        litellm_gateway::router(state, config),
        &litellm_gateway::middleware::Settings::default(),
        Arc::new(Entitlements::Community),
        &litellm_gateway::middleware::InFlight::new(),
        None,
    )
}

async fn call(app: &axum::Router) -> u16 {
    let request = Request::post("/v1/messages")
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "claude", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let _ = axum::body::to_bytes(response.into_body(), 1 << 20).await;
    status
}

async fn scrape(app: &axum::Router, request: Request<Body>) -> (u16, Option<String>, Vec<u8>) {
    let response: Response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let encoding = response
        .headers()
        .get("content-encoding")
        .map(|value| value.to_str().unwrap().to_owned());
    let body = axum::body::to_bytes(response.into_body(), 1 << 24)
        .await
        .unwrap();
    (status, encoding, body.to_vec())
}

fn scraped(path: &str) -> Request<Body> {
    Request::get(path)
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .body(Body::empty())
        .unwrap()
}

fn samples<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
    text.lines()
        .filter(|line| line.starts_with(&format!("{name}{{")))
        .collect()
}

/// The scrape once the success log, which runs after the answer goes out, recorded `name`.
async fn once_logged(app: &axum::Router, name: &str) -> String {
    for _ in 0..100 {
        let (status, _, body) = scrape(app, scraped("/metrics/")).await;
        assert_eq!(status, 200);
        let text = String::from_utf8(body).unwrap();
        if !samples(&text, name).is_empty() {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("{name} was never recorded");
}

fn value(line: &str) -> &str {
    line.rsplit(' ').next().unwrap()
}

#[rstest]
#[tokio::test]
async fn a_served_call_is_counted_by_the_deployment_that_served_it() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-unlisted-9",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5}
        })))
        .mount(&upstream)
        .await;
    let app = app(&config(&upstream)).await;

    assert_eq!(call(&app).await, 200);
    let text = once_logged(&app, "litellm_proxy_total_requests_metric_total").await;

    let requests = samples(&text, "litellm_proxy_total_requests_metric_total");
    assert_eq!(requests.len(), 1, "{text}");
    assert!(
        requests[0].contains("hashed_api_key=\"litellm_proxy_master_key\""),
        "{text}"
    );
    assert!(requests[0].contains("model_id=\"claude-1\""), "{text}");
    assert_eq!(
        samples(&text, "litellm_requests_metric_total")
            .iter()
            .filter(|line| line.contains("model=\"claude-unlisted-9\""))
            .count(),
        1,
        "{text}"
    );
    assert_eq!(value(requests[0]), "1.0");
    let input = samples(&text, "litellm_input_tokens_metric_total");
    assert_eq!(
        input.iter().map(|line| value(line)).collect::<Vec<_>>(),
        ["3.0"],
        "{text}"
    );
    let output = samples(&text, "litellm_output_tokens_metric_total");
    assert_eq!(
        output.iter().map(|line| value(line)).collect::<Vec<_>>(),
        ["5.0"],
        "{text}"
    );
    let deployment = samples(&text, "litellm_deployment_success_responses_total");
    assert_eq!(deployment.len(), 1, "{text}");
    assert!(
        deployment[0].contains(&format!("api_base=\"{}\"", upstream.uri())),
        "{text}"
    );
    assert_eq!(
        samples(&text, "litellm_llm_api_latency_metric_count").len(),
        1,
        "{text}"
    );
}

#[rstest]
#[tokio::test]
async fn a_failed_call_is_counted_as_a_failed_llm_call() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error", "error": {"type": "invalid_request_error", "message": "bad"}
        })))
        .mount(&upstream)
        .await;
    let app = app(&config(&upstream)).await;

    assert_eq!(call(&app).await, 400);
    let (_, _, body) = scrape(&app, scraped("/metrics/")).await;
    let text = String::from_utf8(body).unwrap();

    let failed = samples(&text, "litellm_llm_api_failed_requests_metric_total");
    assert_eq!(
        failed.iter().map(|line| value(line)).collect::<Vec<_>>(),
        ["1.0"],
        "{text}"
    );
    assert!(
        samples(&text, "litellm_requests_metric_total").is_empty(),
        "{text}"
    );
}

#[rstest]
#[case::no_key(None)]
#[case::a_wrong_key(Some("Bearer not-the-key"))]
#[tokio::test]
async fn a_scrape_the_auth_refuses_is_answered_401_as_the_proxy_answers_it(
    #[case] authorization: Option<&str>,
) {
    let upstream = MockServer::start().await;
    let app = app(&config(&upstream)).await;
    let request = authorization
        .into_iter()
        .fold(Request::get("/metrics/"), |request, value| {
            request.header("authorization", value)
        })
        .body(Body::empty())
        .unwrap();

    let (status, _, body) = scrape(&app, request).await;
    let message: String = serde_json::from_slice(&body).unwrap();

    assert_eq!(status, 401);
    assert!(
        message.starts_with("Unauthorized access to metrics endpoint: "),
        "{message}"
    );
    assert!(message.ends_with(
        "To allow unauthenticated access, set `litellm_settings.require_auth_for_metrics_endpoint: false` in your proxy_config.yaml."
    ));
}

#[rstest]
#[tokio::test]
async fn a_scrape_is_restricted_by_name_and_gzipped_when_the_client_accepts_it() {
    let upstream = MockServer::start().await;
    let app = app(&config(&upstream)).await;
    let request = Request::get("/metrics/?name[]=litellm_total_users")
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .header("accept-encoding", "deflate, gzip;q=0.5")
        .body(Body::empty())
        .unwrap();

    let (status, encoding, body) = scrape(&app, request).await;
    let mut text = String::new();
    GzDecoder::new(body.as_slice())
        .read_to_string(&mut text)
        .unwrap();

    assert_eq!((status, encoding.as_deref()), (200, Some("gzip")));
    assert_eq!(
        text,
        "# HELP litellm_total_users Total number of users in LiteLLM\n# TYPE litellm_total_users gauge\nlitellm_total_users 0.0\n"
    );
}

#[rstest]
#[tokio::test]
async fn a_scrape_without_the_slash_is_redirected_to_the_mount() {
    let upstream = MockServer::start().await;
    let app = app(&config(&upstream)).await;
    let request = Request::get("/metrics?name[]=x")
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .header("host", "gateway:4000")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(
        (
            response.status().as_u16(),
            response
                .headers()
                .get("location")
                .unwrap()
                .to_str()
                .unwrap()
        ),
        (307, "http://gateway:4000/metrics/?name[]=x")
    );
}

#[rstest]
#[tokio::test]
async fn no_metrics_are_served_when_the_config_does_not_list_the_logger() {
    let upstream = MockServer::start().await;
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: {MASTER_KEY}\n"
    ))
    .unwrap();
    let _ = upstream;
    let app = app(&config).await;

    let (status, _, _) = scrape(&app, scraped("/metrics/")).await;

    assert_eq!(status, 404);
}
