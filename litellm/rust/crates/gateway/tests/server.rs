use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use axum::{body::Body, http::Request};
use litellm_config::{Config, Environment};
use litellm_db::{Endpoint, PoolLimits, Pools, SslMode};
use litellm_gateway::{AppState, Inference, RedisClients, Resources};
use litellm_gateway_inference::Error;
use litellm_http::ClientVariant;
use litellm_license::Entitlements;
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
use litellm_tracing::{Logger, Metadata, Record, Sink};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

struct LogSink(mpsc::Sender<Value>);

impl Sink for LogSink {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn emit(&self, record: &Record) {
        self.0
            .send(json!({"message": record.message, "fields": record.fields}))
            .unwrap();
    }
}

async fn inference_for(config: &Config) -> Inference {
    litellm_gateway::build_inference(
        config,
        &Environment::default(),
        &Resources::build(config).unwrap(),
        litellm_gateway_inference::RoutingCache::new(
            None,
            litellm_gateway_inference::RouterTiers::default(),
        ),
        &RedisClients::default(),
        None,
    )
    .await
    .unwrap()
}

fn global(router: axum::Router) -> axum::Router {
    litellm_gateway::middleware::global(
        router,
        &litellm_gateway::middleware::Settings::default(),
        Arc::new(Entitlements::Community),
        &litellm_gateway::middleware::InFlight::new(),
        None,
    )
}

#[fixture]
async fn inference() -> Inference {
    inference_for(&Config::from_yaml("model_list: []").unwrap()).await
}

fn without_stores(inference: Inference) -> AppState {
    AppState {
        key_limiter: inference.key_limiter,
        inference: inference.gateway,
        redis: RedisClients::default(),
        database: None,
        keys: None,
        spend_logs: None,
        entitlements: Arc::new(Entitlements::Community),
        environment: Arc::new(litellm_env::ProcessEnvironment),
        counters: litellm_gateway::parallel_request_limiter::counters(None),
        variables: Environment::default(),
        gateway_requests: None,
        gateway_requests_flush: None,
    }
}

#[rstest]
#[case::authorized("/v1/messages", &[("authorization", "Bearer gateway-key")], Some("gateway-key"), 400, None)]
#[case::x_api_key("/v1/messages", &[("x-api-key", "gateway-key")], Some("gateway-key"), 400, None)]
#[case::wrong_x_api_key("/v1/messages", &[("x-api-key", "wrong")], Some("gateway-key"), 401, Some("missing or invalid API key"))]
#[case::missing_token("/v1/messages", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::wrong_token("/v1/messages", &[("authorization", "Bearer wrong")], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::ocr("/ocr", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::unknown_path("/unknown", &[], Some("gateway-key"), 404, None)]
#[case::unknown_path_unconfigured("/unknown", &[], None, 404, None)]
#[case::unconfigured("/v1/messages", &[("authorization", "Bearer gateway-key")], None, 500, None)]
#[tokio::test]
async fn authenticates_before_serving_mounted_inference_routes(
    #[future(awt)] inference: Inference,
    #[case] path: &str,
    #[case] headers: &[(&str, &str)],
    #[case] master_key: Option<&str>,
    #[case] status: u16,
    #[case] unauthorized: Option<&str>,
) {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: {}\n",
        master_key.unwrap_or("null")
    ))
    .unwrap();
    let client = inference
        .gateway
        .resources
        .pool
        .client(&inference.gateway.http, ClientVariant::Provider)
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, stopped) = oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            global(litellm_gateway::router(without_stores(inference), &config)),
        )
        .with_graceful_shutdown(async move {
            let _ = stopped.await;
        })
        .await
    });

    let request = client
        .post(format!("http://{address}{path}"))
        .timeout(Duration::from_secs(5))
        .header("x-request-id", "gateway-test")
        .json(&json!({"model": "unconfigured-model"}));
    let request = headers.iter().fold(request, |request, (name, value)| {
        request.header(*name, *value)
    });
    let response = request.send().await.unwrap();
    assert_eq!(response.status().as_u16(), status);
    if status == 400 {
        let expected = Error::UnknownModel {
            route: "/v1/messages",
            model: "unconfigured-model".into(),
        };
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            expected.body(Some("gateway-test"))
        );
    } else {
        let text = response.text().await.unwrap();
        assert!(!text.contains("gateway-key"));
        assert!(!text.contains("unconfigured-model"));
        assert_eq!(unauthorized.is_some(), status == 401);
        if let Some(message) = unauthorized {
            assert_eq!(text, message);
        }
    }

    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn logs_request_outcome_without_credentials_or_query(#[future(awt)] inference: Inference) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages?token=query-secret")
        .header("authorization", "Bearer header-secret")
        .body(Body::empty())
        .unwrap();
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(LogSink(sender));

    let response = logger
        .instrument(
            global(litellm_gateway::router(without_stores(inference), &config)).oneshot(request),
        )
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 401);
    let record = receiver.try_recv().unwrap();
    assert_eq!(record["message"], "response headers");
    assert_eq!(record["fields"]["method"], "POST");
    assert_eq!(record["fields"]["path"], "/v1/messages");
    assert_eq!(record["fields"]["status"], 401);
    assert!(record["fields"]["time_to_headers_ms"].as_f64().unwrap() >= 0.0);
    assert!(record["fields"]["request_id"].as_str().is_some());
    assert!(receiver.try_recv().is_err());
    assert!(!record.to_string().contains("header-secret"));
    assert!(!record.to_string().contains("query-secret"));
}

#[rstest]
#[case::with_master_key("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")]
#[case::without_master_key("model_list: []\n")]
#[tokio::test]
async fn readiness_answers_without_credentials_like_the_proxy_with_no_database(
    #[future(awt)] inference: Inference,
    #[case] yaml: &str,
) {
    let config = Config::from_yaml(yaml).unwrap();
    let app = global(litellm_gateway::router(without_stores(inference), &config));

    let response = app
        .oneshot(
            Request::get("/health/readiness")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 200);
    let body = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap(),
        json!({"status": "healthy", "db": "Not connected"})
    );
}

#[rstest]
#[tokio::test]
async fn readiness_reports_a_configured_database_it_cannot_reach(
    #[future(awt)] inference: Inference,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_port = listener.local_addr().unwrap().port();
    drop(listener);
    let unreachable = Endpoint {
        host: "127.0.0.1".to_owned(),
        port: closed_port,
        socket: None,
        username: "app".to_owned(),
        password: None,
        database: "proxy".to_owned(),
        ssl_mode: SslMode::Disable,
        ssl_root_cert: None,
        ssl_client: None,
        application_name: None,
        options: Vec::new(),
        statement_cache_capacity: 0,
        pool: PoolLimits {
            max_connections: 1,
            acquire_timeout_seconds: Some(0.2),
            idle_timeout_seconds: None,
            socket_timeout_seconds: None,
        },
    };
    let state = AppState {
        database: Some(Pools::new(&unreachable, None).unwrap()),
        ..without_stores(inference)
    };
    let config = Config::from_yaml("model_list: []").unwrap();

    let response = global(litellm_gateway::router(state, &config))
        .oneshot(
            Request::get("/health/readiness")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 200);
    let body = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap(),
        json!({"status": "healthy", "db": "disconnected"})
    );
}

fn served_message() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-unlisted-9",
        "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
}

fn deployment(name: &str, id: &str, upstream: &MockServer) -> String {
    format!(
        "  - model_name: {name}\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n    model_info:\n      id: {id}\n",
        upstream.uri()
    )
}

fn asking(body: &Value) -> Request<Body> {
    Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn app(config: &str) -> axum::Router {
    let config = Config::from_yaml(config).unwrap();
    global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ))
}

async fn routed(config: &str, model: &str) -> axum::response::Response {
    let body =
        json!({"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});
    app(config).await.oneshot(asking(&body)).await.unwrap()
}

/// The headers that tell what the router did for a call.
const ROUTING_HEADERS: [&str; 5] = [
    "x-litellm-model-id",
    "x-litellm-model-group",
    "x-litellm-attempted-retries",
    "x-litellm-attempted-fallbacks",
    "x-litellm-max-retries",
];

fn routing_headers(response: &axum::response::Response) -> Vec<(&str, &str)> {
    response
        .headers()
        .iter()
        .filter(|(name, _)| ROUTING_HEADERS.contains(&name.as_str()))
        .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
        .collect()
}

#[rstest]
#[tokio::test]
async fn a_failed_model_group_falls_back_to_the_configured_group() {
    let (failing, healthy) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "refused"}})),
        )
        .expect(1)
        .mount(&failing)
        .await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .expect(1)
        .mount(&healthy)
        .await;
    let config = format!(
        "model_list:\n{}{}router_settings:\n  fallbacks:\n    - primary: [backup]\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &failing),
        deployment("backup", "backup-1", &healthy),
    );

    let response = routed(&config, "primary").await;

    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        routing_headers(&response),
        [
            ("x-litellm-model-id", "backup-1"),
            ("x-litellm-model-group", "backup"),
            ("x-litellm-attempted-retries", "0"),
            ("x-litellm-attempted-fallbacks", "1"),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn a_name_no_deployment_serves_routes_to_the_first_default_fallback() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .expect(1)
        .mount(&upstream)
        .await;
    let config = format!(
        "model_list:\n{}router_settings:\n  default_fallbacks: [backup]\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("backup", "backup-1", &upstream),
    );

    let response = routed(&config, "unconfigured").await;

    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        routing_headers(&response)[0],
        ("x-litellm-model-id", "backup-1")
    );
}

#[rstest]
#[tokio::test]
async fn a_failed_deployment_is_retried_on_another_deployment_of_its_group() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"error": {"message": "down"}})),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .with_priority(2)
        .mount(&upstream)
        .await;
    let config = format!(
        "model_list:\n{}{}general_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &upstream),
        deployment("primary", "primary-2", &upstream),
    );

    let response = routed(&config, "primary").await;

    assert_eq!(response.status().as_u16(), 200);
    let headers = routing_headers(&response);
    assert_eq!(
        (
            &headers[1..],
            upstream.received_requests().await.unwrap().len()
        ),
        (
            [
                ("x-litellm-model-group", "primary"),
                ("x-litellm-attempted-retries", "1"),
                ("x-litellm-attempted-fallbacks", "0"),
                ("x-litellm-max-retries", "2"),
            ]
            .as_slice(),
            2,
        )
    );
}

#[rstest]
#[tokio::test]
async fn a_deployments_own_num_retries_retries_its_failed_call() {
    let (failing, healthy) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"error": {"message": "down"}})),
        )
        .mount(&failing)
        .await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&healthy)
        .await;
    let deployment = |id: &str, upstream: &MockServer, weight: u8| {
        format!(
            "  - model_name: primary\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n      weight: {weight}\n      num_retries: 1\n    model_info:\n      id: {id}\n",
            upstream.uri()
        )
    };

    let response = routed(
        &format!(
            "model_list:\n{}{}router_settings:\n  num_retries: 0\n  allowed_fails: 0\n  cooldown_time: 30\ngeneral_settings:\n  master_key: gateway-key\n",
            deployment("first", &failing, 1),
            deployment("second", &healthy, 0),
        ),
        "primary",
    )
    .await;

    assert_eq!(
        (response.status().as_u16(), routing_headers(&response)),
        (
            200,
            vec![
                ("x-litellm-model-id", "second"),
                ("x-litellm-model-group", "primary"),
                ("x-litellm-attempted-retries", "1"),
                ("x-litellm-attempted-fallbacks", "0"),
                ("x-litellm-max-retries", "1"),
            ]
        )
    );
}

/// A retry policy names litellm's error classes, so a provider's status has to land in the class
/// the Python proxy puts it in: an overloaded Anthropic answer is an `InternalServerError`.
#[rstest]
#[case::it_is_an_internal_server_error("InternalServerErrorRetries", 200)]
#[case::it_is_not_an_unavailable_service("ServiceUnavailableErrorRetries", 529)]
#[tokio::test]
async fn a_retry_policy_retries_an_overloaded_provider_by_its_error_class(
    #[case] retried_class: &str,
    #[case] answered: u16,
) {
    let (failing, healthy) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(529).set_body_json(
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        ))
        .mount(&failing)
        .await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&healthy)
        .await;

    let response = routed(
        &format!(
            "model_list:\n{}{}router_settings:\n  num_retries: 0\n  allowed_fails: 0\n  cooldown_time: 30\n  retry_policy:\n    {retried_class}: 1\ngeneral_settings:\n  master_key: gateway-key\n",
            weighted("primary", "first", &failing, 1),
            weighted("primary", "second", &healthy, 0),
        ),
        "primary",
    )
    .await;

    assert_eq!(response.status().as_u16(), answered);
}

#[tokio::test]
async fn a_call_through_an_alias_reports_the_alias_as_its_model_group() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&upstream)
        .await;
    let config = format!(
        "model_list:\n{}router_settings:\n  model_group_alias:\n    alias: primary\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &upstream),
    );

    let response = routed(&config, "alias").await;

    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        routing_headers(&response)[..2],
        [
            ("x-litellm-model-id", "primary-1"),
            ("x-litellm-model-group", "alias"),
        ]
    );
}

/// The proxy reads a group's usage by the group an alias names, and still reports the alias as the
/// model group.
#[tokio::test]
async fn a_call_through_an_alias_tells_the_remaining_usage_of_the_group_it_names() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&upstream)
        .await;
    let config = format!(
        "model_list:\n  - model_name: primary\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n      rpm: 10\n    model_info:\n      id: primary-1\nrouter_settings:\n  model_group_alias:\n    alias: primary\ngeneral_settings:\n  master_key: gateway-key\n",
        upstream.uri()
    );

    let response = routed(&config, "alias").await;

    let told = |name: &str| response.headers().get(name)?.to_str().ok();
    assert_eq!(
        (
            told("x-litellm-model-group"),
            told("x-ratelimit-limit-requests"),
            told("x-ratelimit-remaining-requests").is_some(),
        ),
        (Some("alias"), Some("10"), true)
    );
}

#[rstest]
#[tokio::test]
async fn a_call_that_ends_on_a_failed_attempt_names_its_deployment() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "refused"}})),
        )
        .mount(&upstream)
        .await;
    let config = format!(
        "model_list:\n{}general_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &upstream),
    );

    let response = routed(&config, "primary").await;

    assert_eq!(response.status().as_u16(), 400);
    assert_eq!(
        routing_headers(&response),
        [("x-litellm-model-id", "primary-1")]
    );
}

#[rstest]
#[tokio::test]
async fn a_call_whose_fallback_also_fails_names_the_deployment_whose_error_it_answers_with() {
    let (first, second) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"error": {"message": "from-first"}})),
        )
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(503).set_body_json(json!({"error": {"message": "from-second"}})),
        )
        .mount(&second)
        .await;
    let config = format!(
        "model_list:\n{}{}router_settings:\n  num_retries: 0\n  fallbacks:\n    - first: [second]\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("first", "first-1", &first),
        deployment("second", "second-1", &second),
    );

    let response = routed(&config, "first").await;

    assert_eq!(
        (response.status().as_u16(), routing_headers(&response)),
        (500, vec![("x-litellm-model-id", "first-1")])
    );
    assert_eq!(second.received_requests().await.unwrap().len(), 1);
}

/// A port nothing listens on, so a call to it never gets an answer.
async fn closed_port() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    format!("http://{}", listener.local_addr().unwrap())
}

#[tokio::test]
async fn a_deployment_the_gateway_cannot_reach_cools_down_and_its_group_keeps_serving() {
    let healthy = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&healthy)
        .await;
    let unreachable = closed_port().await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: primary\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {unreachable}\n      api_key: test-key\n      weight: 1\n    model_info:\n      id: unreachable\n  - model_name: primary\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n      weight: 0\n    model_info:\n      id: healthy\nrouter_settings:\n  num_retries: 0\n  cooldown_time: 30\ngeneral_settings:\n  master_key: gateway-key\n",
        healthy.uri()
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));
    let call = || {
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model": "primary", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                    .to_string(),
            ))
            .unwrap();
        app.clone().oneshot(request)
    };

    let failed = [
        call().await.unwrap().status().as_u16(),
        call().await.unwrap().status().as_u16(),
        call().await.unwrap().status().as_u16(),
        call().await.unwrap().status().as_u16(),
        call().await.unwrap().status().as_u16(),
    ];
    let served = call().await.unwrap();

    assert_eq!(failed, [500; 5]);
    assert_eq!(
        (
            served.status().as_u16(),
            served.headers()["x-litellm-model-id"].to_str().unwrap()
        ),
        (200, "healthy")
    );
}

/// router-core checks no setting when it builds, so a strategy the native host does not pick by
/// fails each call that reaches it, where the provider is never asked.
#[tokio::test]
async fn a_strategy_the_native_host_does_not_pick_by_fails_the_call() {
    let provider = MockServer::start().await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\nrouter_settings:\n  routing_strategy: usage-based-routing-v2\ngeneral_settings:\n  master_key: gateway-key\n",
        provider.uri()
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "public-model", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status().as_u16(), 500);
    assert!(provider.received_requests().await.unwrap().is_empty());
}

async fn answering() -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&upstream)
        .await;
    upstream
}

async fn reached(upstream: &MockServer) -> bool {
    !upstream.received_requests().await.unwrap().is_empty()
}

/// A `claude` group of a deployment with a 100 token window and one of `large_model` with a 1000
/// token window, with the pre-call checks on.
fn two_windows(small: &MockServer, large: &MockServer, large_model: &str) -> String {
    let sized = |id: &str, model: &str, upstream: &MockServer, window: u32| {
        format!(
            "  - model_name: claude\n    litellm_params:\n      model: anthropic/{model}\n      api_base: {}\n      api_key: test-key\n    model_info:\n      id: {id}\n      max_input_tokens: {window}\n",
            upstream.uri()
        )
    };
    format!(
        "model_list:\n{}{}router_settings:\n  enable_pre_call_checks: true\ngeneral_settings:\n  master_key: gateway-key\n",
        sized("small", UNLISTED, small, 100),
        sized("large", large_model, large, 1000),
    )
}

const UNLISTED: &str = "in-house-9";

/// One user message of `words` words, a few tokens over `words` as the gateway estimates it.
fn prompt(words: usize) -> Value {
    json!({
        "model": "claude", "max_tokens": 8,
        "messages": [{"role": "user", "content": "word ".repeat(words)}],
    })
}

#[rstest]
#[case::every_deployment_it_fits(10, (true, true))]
#[case::only_the_deployments_it_fits(150, (false, true))]
#[case::a_deployment_its_estimate_overflows_by_less_than_twice_when_none_fits(1500, (false, true))]
#[tokio::test]
async fn a_prompt_reaches(#[case] words: usize, #[case] expected: (bool, bool)) {
    let (small, large) = (answering().await, answering().await);
    let app = app(&two_windows(&small, &large, UNLISTED)).await;

    for _ in 0..30 {
        let response = app.clone().oneshot(asking(&prompt(words))).await.unwrap();
        assert_eq!(response.status().as_u16(), 200);
    }

    assert_eq!((reached(&small).await, reached(&large).await), expected);
}

#[rstest]
#[case::past_twice_every_window(3000, UNLISTED)]
#[case::over_a_window_its_own_model_info_caps_below_the_cost_maps(1500, "claude-haiku-5-5")]
#[tokio::test]
async fn a_prompt_no_deployment_fits_is_refused_before_any_provider_call(
    #[case] words: usize,
    #[case] large_model: &str,
) {
    let (small, large) = (answering().await, answering().await);
    let app = app(&two_windows(&small, &large, large_model)).await;

    let response = app.oneshot(asking(&prompt(words))).await.unwrap();

    let status = response.status().as_u16();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let answer: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        (
            status,
            answer["error"]["type"].as_str(),
            reached(&small).await,
            reached(&large).await
        ),
        (400, Some("invalid_request_error"), false, false)
    );
}

fn streamed(events: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(events.to_owned(), "text/event-stream")
}

async fn stream_read_to_its_end(app: &axum::Router, model: &str) -> (u16, String) {
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": model, "max_tokens": 8, "stream": true, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

/// An error event counts against its deployment as the same error does before a stream opens.
/// The Python proxy records such a stream as served when no attempt can answer in its place.
#[rstest]
#[case::an_overloaded_provider_is_cooled_down(OVERLOADED, [200, 200, 429], 2)]
#[case::a_refused_request_is_not(REFUSED, [200, 200, 200], 3)]
#[tokio::test]
async fn a_stream_that_goes_out_with_the_providers_error_counts_against_its_deployment_by_that_error(
    #[case] error: &str,
    #[case] answered: [u16; 3],
    #[case] asked: usize,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(streamed(&[MESSAGE_START, error].concat()))
        .mount(&upstream)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}router_settings:\n  allowed_fails: 0\n  cooldown_time: 30\n  num_retries: 0\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &upstream),
        deployment("primary", "primary-2", &upstream),
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));

    let statuses = [
        stream_read_to_its_end(&app, "primary").await.0,
        stream_read_to_its_end(&app, "primary").await.0,
        stream_read_to_its_end(&app, "primary").await.0,
    ];

    assert_eq!(
        (statuses, upstream.received_requests().await.unwrap().len()),
        (answered, asked)
    );
}

#[tokio::test]
async fn a_failed_stream_cools_its_deployment_down_for_the_deployments_own_cooldown_time() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(streamed(&[MESSAGE_START, OVERLOADED].concat()))
        .mount(&upstream)
        .await;
    let cooling_for_7s = |id: &str| {
        format!(
            "  - model_name: primary\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n      cooldown_time: 7\n    model_info:\n      id: {id}\n",
            upstream.uri()
        )
    };
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}router_settings:\n  allowed_fails: 0\n  cooldown_time: 30\n  num_retries: 0\ngeneral_settings:\n  master_key: gateway-key\n",
        cooling_for_7s("primary-1"),
        cooling_for_7s("primary-2"),
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));

    stream_read_to_its_end(&app, "primary").await;
    stream_read_to_its_end(&app, "primary").await;
    let (status, refused) = stream_read_to_its_end(&app, "primary").await;

    assert!(
        status == 429 && refused.contains("Try again in 7 seconds"),
        "{status} {refused}"
    );
}

#[tokio::test]
async fn a_stream_that_ends_cleanly_keeps_its_deployment_serving() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(streamed(concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-unlisted-9\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        )))
        .mount(&upstream)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}router_settings:\n  allowed_fails: 0\n  cooldown_time: 30\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &upstream),
        deployment("primary", "primary-2", &upstream),
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));

    let statuses = [
        stream_read_to_its_end(&app, "primary").await.0,
        stream_read_to_its_end(&app, "primary").await.0,
        stream_read_to_its_end(&app, "primary").await.0,
    ];

    assert_eq!(statuses, [200, 200, 200]);
}

const MESSAGE_START: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-unlisted-9\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
const TEXT: &str = concat!(
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
);
const MESSAGE_END: &str = concat!(
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);
const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";

fn weighted(name: &str, id: &str, upstream: &MockServer, weight: u8) -> String {
    format!(
        "  - model_name: {name}\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n      weight: {weight}\n    model_info:\n      id: {id}\n",
        upstream.uri()
    )
}

/// A router whose `primary` group is first served by a deployment that answers a stream with
/// `failing`, and whose `second` deployment, of `second_group`, answers one that says "hi".
async fn streamed_app(
    failing: &str,
    second_group: &str,
    settings: &str,
) -> (axum::Router, MockServer, MockServer) {
    let answering = [MESSAGE_START, TEXT, MESSAGE_END].concat();
    streamed_pair(failing, &answering, second_group, settings).await
}

/// A router whose `primary` group is first served by a deployment that answers a stream with
/// `first_sends`, and whose `second` deployment, of `second_group`, answers one with
/// `second_sends`.
async fn streamed_pair(
    first_sends: &str,
    second_sends: &str,
    second_group: &str,
    settings: &str,
) -> (axum::Router, MockServer, MockServer) {
    let (first, second) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(streamed(first_sends))
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(streamed(second_sends))
        .mount(&second)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}router_settings:\n{settings}general_settings:\n  master_key: gateway-key\n",
        weighted("primary", "first", &first, 1),
        weighted(second_group, "second", &second, 0),
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));
    (app, first, second)
}

fn stream_of(model: &str) -> Request<Body> {
    Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": model, "max_tokens": 8, "stream": true, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap()
}

#[rstest]
#[case::retried_in_its_group(
    "primary",
    "  num_retries: 1\n  allowed_fails: 0\n  cooldown_time: 30\n",
    vec![
        ("x-litellm-model-id", "second"),
        ("x-litellm-model-group", "primary"),
        ("x-litellm-attempted-retries", "1"),
        ("x-litellm-attempted-fallbacks", "0"),
        ("x-litellm-max-retries", "1"),
    ]
)]
#[case::served_by_a_fallback(
    "backup",
    "  num_retries: 0\n  fallbacks:\n    - primary: [backup]\n",
    vec![
        ("x-litellm-model-id", "second"),
        ("x-litellm-model-group", "backup"),
        ("x-litellm-attempted-retries", "0"),
        ("x-litellm-attempted-fallbacks", "1"),
    ]
)]
#[tokio::test]
async fn a_stream_the_provider_fails_before_any_content_is_served_by_another_attempt(
    #[case] second_group: &str,
    #[case] settings: &str,
    #[case] routed: Vec<(&str, &str)>,
) {
    let (app, first, second) = streamed_app(
        &[MESSAGE_START, OVERLOADED].concat(),
        second_group,
        settings,
    )
    .await;

    let response = app.oneshot(stream_of("primary")).await.unwrap();

    let status = response.status().as_u16();
    assert_eq!(routing_headers(&response), routed);
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let body = std::str::from_utf8(&body).unwrap();
    assert_eq!(
        (
            status,
            body.matches("event: message_start").count(),
            body.contains("\"text\":\"hi\""),
            body.contains("overloaded_error"),
            first.received_requests().await.unwrap().len(),
            second.received_requests().await.unwrap().len(),
        ),
        (200, 1, true, false, 1, 1),
        "{body}"
    );
}

/// The stream that answers in a failed stream's place is held like the first one. A fallback
/// whose own stream fails before content has no fallback of its own left, so with no retry
/// either its stream goes out with the error its provider sent.
#[tokio::test]
async fn a_fallbacks_stream_that_fails_too_goes_out_with_its_providers_error() {
    let failing = [MESSAGE_START, OVERLOADED].concat();
    let (app, first, second) = streamed_pair(
        &failing,
        &failing,
        "backup",
        "  num_retries: 0\n  fallbacks:\n    - primary: [backup]\n",
    )
    .await;

    let response = app.oneshot(stream_of("primary")).await.unwrap();

    let status = response.status().as_u16();
    assert_eq!(
        routing_headers(&response),
        [
            ("x-litellm-model-id", "second"),
            ("x-litellm-model-group", "backup"),
            ("x-litellm-attempted-retries", "0"),
            ("x-litellm-attempted-fallbacks", "1"),
        ]
    );
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let body = std::str::from_utf8(&body).unwrap();
    assert_eq!(
        (
            status,
            body.matches("event: ").count(),
            body.matches("overloaded_error").count(),
            first.received_requests().await.unwrap().len(),
            second.received_requests().await.unwrap().len(),
        ),
        (200, failing.matches("event: ").count(), 1, 1, 1),
        "{body}"
    );
}

const TIMED_OUT: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"timeout_error\",\"message\":\"Slow\"}}\n\n";

#[rstest]
#[case::overloaded(OVERLOADED, 529, "overloaded_error", "Overloaded")]
#[case::timed_out(TIMED_OUT, 504, "timeout_error", "Slow")]
#[tokio::test]
async fn a_stream_whose_take_over_serves_nothing_answers_the_providers_error(
    #[case] failure: &str,
    #[case] status: u16,
    #[case] error_type: &str,
    #[case] message: &str,
) {
    let (first, second) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(streamed(&[MESSAGE_START, failure].concat()))
        .mount(&first)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"error": {"message": "down"}})),
        )
        .mount(&second)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}router_settings:\n  num_retries: 0\n  fallbacks:\n    - primary: [backup]\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("primary", "first", &first),
        deployment("backup", "second", &second),
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));

    let response = app.oneshot(stream_of("primary")).await.unwrap();

    let answered = response.status().as_u16();
    assert_eq!(
        routing_headers(&response),
        [
            ("x-litellm-model-id", "first"),
            ("x-litellm-model-group", "primary"),
            ("x-litellm-attempted-retries", "0"),
            ("x-litellm-attempted-fallbacks", "0"),
        ]
    );
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    assert_eq!(
        (answered, serde_json::from_slice::<Value>(&body).unwrap()),
        (
            status,
            json!({"type": "error", "error": {"type": error_type, "message": message}})
        )
    );
}

const REFUSED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"Refused\"}}\n\n";

const RETRIED_AND_FALLEN_BACK: &str = "  num_retries: 1\n  fallbacks:\n    - primary: [backup]\n";

#[rstest]
#[case::once_content_was_sent(
    &[MESSAGE_START, TEXT, OVERLOADED],
    RETRIED_AND_FALLEN_BACK,
    "overloaded_error"
)]
#[case::on_an_error_no_retry_clears(
    &[MESSAGE_START, REFUSED],
    RETRIED_AND_FALLEN_BACK,
    "invalid_request_error"
)]
#[case::when_the_retry_policy_gives_its_error_class_no_retry(
    &[MESSAGE_START, OVERLOADED],
    "  num_retries: 0\n  retry_policy:\n    RateLimitErrorRetries: 1\n",
    "overloaded_error"
)]
#[tokio::test]
async fn a_stream_no_other_attempt_may_answer_goes_out_as_the_provider_sent_it(
    #[case] failing: &[&str],
    #[case] settings: &str,
    #[case] error: &str,
) {
    let sent = failing.concat();
    let (app, first, second) = streamed_app(&sent, "backup", settings).await;

    let response = app.oneshot(stream_of("primary")).await.unwrap();

    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let body = std::str::from_utf8(&body).unwrap();
    assert_eq!(
        (
            body.matches("event: ").count(),
            body.contains(error),
            first.received_requests().await.unwrap().len(),
            second.received_requests().await.unwrap().len(),
        ),
        (sent.matches("event: ").count(), true, 1, 0),
        "{body}"
    );
}

const SERVED_STREAM: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-unlisted-9\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

/// The model a `/v1/messages` answer reports, in its body or in its stream's `message_start`.
fn reported_model(body: &str) -> Value {
    let message_start = body
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<Value>(data).unwrap()["message"].clone());
    message_start.unwrap_or_else(|| serde_json::from_str(body).unwrap())["model"].clone()
}

#[rstest]
#[case::served_by_its_own_group("backup", false, "backup", "0")]
#[case::served_by_a_fallback("primary", false, "claude-unlisted-9", "1")]
#[case::a_stream_served_by_its_own_group("backup", true, "backup", "0")]
#[case::a_stream_served_by_a_fallback("primary", true, "primary", "1")]
#[tokio::test]
async fn a_messages_answer_reports_the_model_the_client_asked_for_and_how_it_was_routed(
    #[case] model: &str,
    #[case] stream: bool,
    #[case] expected: &str,
    #[case] fallbacks: &str,
) {
    let (failing, healthy) = (MockServer::start().await, MockServer::start().await);
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "refused"}})),
        )
        .mount(&failing)
        .await;
    Mock::given(method("POST"))
        .respond_with(if stream {
            streamed(SERVED_STREAM)
        } else {
            served_message()
        })
        .mount(&healthy)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}router_settings:\n  fallbacks:\n    - primary: [backup]\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("primary", "primary-1", &failing),
        deployment("backup", "backup-1", &healthy),
    ))
    .unwrap();
    let app = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ));
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": model, "max_tokens": 8, "stream": stream, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    let status = response.status().as_u16();
    assert_eq!(
        routing_headers(&response),
        [
            ("x-litellm-model-id", "backup-1"),
            ("x-litellm-model-group", "backup"),
            ("x-litellm-attempted-retries", "0"),
            ("x-litellm-attempted-fallbacks", fallbacks),
        ]
    );
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    assert_eq!(
        (status, reported_model(std::str::from_utf8(&body).unwrap())),
        (200, json!(expected))
    );
}

async fn forwarded_body(deployment_params: &str, client_params: Value) -> Value {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-unlisted-9",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&upstream)
        .await;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.yaml");
    std::fs::write(&path, format!(
        "environment_variables:\n  SYSTEM_PROMPT: Answer in French.\n  MCP_TOKEN: mcp-token\nmodel_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n{deployment_params}general_settings:\n  master_key: gateway-key\n",
        upstream.uri()
    ))
    .unwrap();
    let config = litellm_gateway::load_config(&path).await.unwrap().config;
    let mut body = json!({
        "model": "public-model",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    });
    body.as_object_mut()
        .unwrap()
        .extend(client_params.as_object().cloned().unwrap_or_default());
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ))
    .oneshot(request)
    .await
    .unwrap();

    assert_eq!(response.status().as_u16(), 200);
    serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body).unwrap()
}

#[rstest]
#[case::system(
    "      system: os.environ/SYSTEM_PROMPT\n",
    "/system",
    json!("Answer in French.")
)]
#[case::mcp_server_token(
    "      mcp_servers:\n        - type: url\n          url: https://mcp.example.com/sse\n          name: docs\n          authorization_token: os.environ/MCP_TOKEN\n",
    "/mcp_servers/0/authorization_token",
    json!("mcp-token")
)]
#[tokio::test]
async fn a_deployment_param_from_the_environment_reaches_the_provider_resolved(
    #[case] deployment_params: &str,
    #[case] pointer: &str,
    #[case] expected: Value,
) {
    let forwarded = forwarded_body(deployment_params, json!({})).await;

    assert_eq!(forwarded.pointer(pointer), Some(&expected));
}

#[rstest]
#[case::listed("      additional_drop_params: [temperature]\n", None)]
#[case::unlisted("", Some(json!(0.5)))]
#[tokio::test]
async fn a_deployment_additional_drop_param_never_reaches_the_provider(
    #[case] deployment_params: &str,
    #[case] expected: Option<Value>,
) {
    let forwarded = forwarded_body(deployment_params, json!({"temperature": 0.5})).await;

    assert_eq!(forwarded.get("temperature").cloned(), expected);
}

#[rstest]
#[tokio::test]
async fn a_requests_messages_settings_lay_over_the_deployments_and_never_reach_the_provider() {
    let forwarded = forwarded_body(
        "      additional_drop_params: [top_k]\n",
        json!({
            "temperature": 0.5,
            "top_k": 5,
            "drop_params": true,
            "additional_drop_params": ["temperature"],
        }),
    )
    .await;

    assert_eq!(
        [
            "temperature",
            "top_k",
            "drop_params",
            "additional_drop_params"
        ]
        .map(|key| forwarded.get(key).cloned()),
        [None, Some(json!(5)), None, None]
    );
}

#[rstest]
#[case::text_python_reads_as_on("  reasoning_auto_summary: \"false\"\n", Some(json!("summarized")))]
#[case::off("  reasoning_auto_summary: false\n", None)]
#[tokio::test]
async fn litellm_settings_reasoning_auto_summary_asks_for_summarized_thinking(
    #[case] settings: &str,
    #[case] expected: Option<Value>,
) {
    let forwarded = forwarded_body(
        &format!("    model_info:\n      supports_reasoning: true\nlitellm_settings:\n{settings}"),
        json!({"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
    )
    .await;

    assert_eq!(forwarded["thinking"].get("display").cloned(), expected);
}

#[rstest]
#[case::fast_mode("supports_speed", "speed", json!("fast"))]
#[case::adaptive_thinking("supports_adaptive_thinking", "thinking", json!({"type": "adaptive"}))]
#[tokio::test]
async fn forwards_a_param_the_catalog_says_the_model_supports(
    #[case] capability: &str,
    #[case] param: &str,
    #[case] value: Value,
) {
    let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
    let (model, _) = catalog
        .served()
        .find(|(name, fields)| {
            !name.contains('/')
                && fields.get("litellm_provider").and_then(Value::as_str) == Some("anthropic")
                && fields.get(capability) == Some(&Value::Bool(true))
        })
        .unwrap();
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&upstream)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/{model}\n      api_base: {}\n      api_key: test-key\ngeneral_settings:\n  master_key: gateway-key\n",
        upstream.uri()
    ))
    .unwrap();
    let body: serde_json::Map<String, Value> = [
        ("model", json!("public-model")),
        ("max_tokens", json!(8)),
        ("messages", json!([{"role": "user", "content": "hi"}])),
        (param, value.clone()),
    ]
    .into_iter()
    .map(|(key, field)| (key.to_owned(), field))
    .collect();
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(Value::Object(body).to_string()))
        .unwrap();

    let response = global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ))
    .oneshot(request)
    .await
    .unwrap();

    assert_eq!(response.status().as_u16(), 200);
    let forwarded: Value =
        serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(forwarded[param], value);
}

const VIRTUAL_KEY: &str = "sk-cached-virtual-key";

/// A key as Python's auth cache holds it, cut down to the fields the gateway cannot do without.
fn python_cached_key() -> Value {
    json!({
        "token": litellm_gateway_auth_core::Token::new(VIRTUAL_KEY).hash().to_string(), "soft_budget_cooldown": false,
        "spend": 0.0, "total_spend": 0.0, "models": [], "aliases": {}, "config": {},
        "permissions": {}, "metadata": {}, "model_spend": {}, "model_max_budget": {},
        "budget_fallbacks": {}, "team_models": [], "team_blocked": false, "expires": null,
        "budget_reset_at": null, "created_at": null, "updated_at": null,
        "settings_updated_at": null, "last_active": null, "last_rotation_at": null,
        "key_rotation_at": null
    })
}

/// A gateway whose database never answers and whose coordination Redis holds `cached` under the
/// key's hash, as Python's `UserApiKeyCache` writes it.
async fn behind_the_auth_cache(
    cached: Option<Value>,
    litellm_settings: &str,
    upstream: &MockServer,
) -> axum::response::Response {
    asking_behind_the_auth_cache(cached, litellm_settings, upstream, "claude").await
}

async fn asking_behind_the_auth_cache(
    cached: Option<Value>,
    litellm_settings: &str,
    upstream: &MockServer,
    model: &str,
) -> axum::response::Response {
    let redis = litellm_redis_testing::RedisServer::start().await.unwrap();
    if let Some(cached) = cached {
        let hash = litellm_gateway_auth_core::Token::new(VIRTUAL_KEY)
            .hash()
            .to_string();
        redis
            .cli(&["SET", &hash, &cached.to_string()])
            .await
            .unwrap();
    }
    let database = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_port = database.local_addr().unwrap().port();
    drop(database);
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}general_settings:\n  master_key: gateway-key\n  database_url: postgresql://app@127.0.0.1:{closed_port}/proxy\n  database_connection_timeout: 0.5\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\nlitellm_settings:\n{litellm_settings}",
        deployment("claude", "claude-a", upstream),
        deployment("claude-down", "claude-down", upstream),
        redis.port()
    ))
    .unwrap();
    let state = AppState::over(
        &config,
        &litellm_config::Environment::default(),
        &Resources::build(&config).unwrap(),
    )
    .await
    .unwrap();
    let request = Request::post("/v1/messages")
        .header("authorization", format!("Bearer {VIRTUAL_KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();
    global(litellm_gateway::router(state, &config))
        .oneshot(request)
        .await
        .unwrap()
}

#[rstest]
#[case::plain(python_cached_key(), 200)]
#[case::blocked(json!({"blocked": true}), 401)]
#[case::with_a_budget(json!({"max_budget": 5.0}), 403)]
#[case::with_a_user_limit(json!({"user_id": "u", "user_rpm_limit": 3}), 403)]
#[case::with_an_end_user_limit(json!({"end_user_id": "e", "end_user_tpm_limit": 3}), 403)]
#[case::with_a_field_the_gateway_does_not_know(json!({"future_limit": 1}), 403)]
#[case::with_an_empty_field_the_gateway_does_not_know(json!({"future_limit": null}), 200)]
#[tokio::test]
async fn a_virtual_key_python_cached_is_judged_without_the_database(
    #[case] change: Value,
    #[case] status: u16,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&upstream)
        .await;
    let mut cached = python_cached_key();
    cached
        .as_object_mut()
        .unwrap()
        .extend(change.as_object().unwrap().clone());

    let response =
        behind_the_auth_cache(Some(cached), "  enable_redis_auth_cache: true\n", &upstream).await;

    assert_eq!(response.status().as_u16(), status);
}

#[rstest]
#[case::cache_off(Some(python_cached_key()), "  enable_redis_auth_cache: false\n")]
#[case::not_cached(None, "  enable_redis_auth_cache: true\n")]
#[case::an_entry_the_gateway_cannot_read(
    Some({
        let mut cached = python_cached_key();
        cached["spend"] = json!("free");
        cached
    }),
    "  enable_redis_auth_cache: true\n"
)]
#[tokio::test]
async fn a_key_the_auth_cache_does_not_answer_is_read_from_the_database(
    #[case] cached: Option<Value>,
    #[case] litellm_settings: &str,
) {
    let upstream = MockServer::start().await;

    let response = behind_the_auth_cache(cached, litellm_settings, &upstream).await;

    assert_eq!(response.status().as_u16(), 503);
}

#[rstest]
#[case::a_model_on_its_list("claude", 200, 1)]
#[case::a_model_off_its_list("claude-down", 403, 0)]
#[tokio::test]
async fn a_virtual_key_with_a_model_list_reaches_only_the_models_it_lists(
    #[case] asked: &str,
    #[case] status: u16,
    #[case] sent: u64,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .expect(sent)
        .mount(&upstream)
        .await;
    let mut cached = python_cached_key();
    cached["models"] = json!(["claude"]);

    let response = asking_behind_the_auth_cache(
        Some(cached),
        "  enable_redis_auth_cache: true\n",
        &upstream,
        asked,
    )
    .await;

    assert_eq!(response.status().as_u16(), status);
}

/// A native answer for `group`, whose deployment `claude-a` carries `limits`, while Python has
/// already counted 100 tokens and 5 requests against it this minute and the next, in the
/// coordination Redis under the router core's `global_router:` counters. Returns the response and
/// each counter's two minutes as Redis then holds them.
async fn counted_with_python(limits: &str) -> (axum::response::Response, Vec<String>) {
    let redis = litellm_redis_testing::RedisServer::start().await.unwrap();
    let now = std::time::SystemTime::now();
    let minutes =
        [now, now + Duration::from_secs(60)].map(litellm_router_core::health::current_minute);
    let counters: Vec<String> = ["tpm", "rpm"]
        .iter()
        .flat_map(|kind| {
            minutes.iter().map(move |minute| {
                format!("global_router:claude-a:anthropic/claude-unlisted-9:{kind}:{minute}")
            })
        })
        .collect();
    for (counter, count) in counters.iter().zip(["100", "100", "5", "5"]) {
        redis.cli(&["SET", counter, count]).await.unwrap();
    }
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(served_message())
        .mount(&upstream)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      api_base: {}\n      api_key: test-key\n{limits}    model_info:\n      id: claude-a\ngeneral_settings:\n  master_key: gateway-key\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\n",
        upstream.uri(),
        redis.port()
    ))
    .unwrap();
    let state = AppState::over(
        &config,
        &litellm_config::Environment::default(),
        &Resources::build(&config).unwrap(),
    )
    .await
    .unwrap();
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "claude", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();
    let response = global(litellm_gateway::router(state, &config))
        .oneshot(request)
        .await
        .unwrap();
    let mut held = Vec::new();
    for counter in &counters {
        held.push(
            redis
                .cli(&["GET", counter])
                .await
                .unwrap()
                .trim()
                .to_owned(),
        );
    }
    (response, held)
}

const GROUP_LIMIT_HEADERS: [&str; 4] = [
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-requests",
];

#[rstest]
#[case::limited(
    "      tpm: 1000\n      rpm: 10\n",
    vec![("x-ratelimit-remaining-tokens", "898"), ("x-ratelimit-limit-tokens", "1000"),
        ("x-ratelimit-remaining-requests", "4"), ("x-ratelimit-limit-requests", "10")]
)]
#[case::requests_only("      rpm: 10\n", vec![("x-ratelimit-remaining-requests", "4"), ("x-ratelimit-limit-requests", "10")])]
#[case::unlimited("", vec![])]
#[tokio::test]
async fn a_native_answer_tells_its_model_groups_remaining_usage_counted_with_python(
    #[case] limits: &str,
    #[case] headers: Vec<(&str, &str)>,
) {
    let (response, held) = counted_with_python(limits).await;

    let told: Vec<(&str, &str)> = GROUP_LIMIT_HEADERS
        .into_iter()
        .filter_map(|name| Some((name, response.headers().get(name)?.to_str().ok()?)))
        .collect();
    assert_eq!((response.status().as_u16(), told), (200, headers));
    let (tokens, requests) = held.split_at(2);
    let counted = !limits.is_empty();
    assert_eq!(
        (
            tokens.contains(&"102".to_owned()),
            requests.contains(&"6".to_owned())
        ),
        (counted, counted),
        "{held:?}"
    );
}
