use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use axum::{body::Body, http::Request};
use litellm_config::Config;
use litellm_db::{Endpoint, PoolLimits, Pools, SslMode};
use litellm_gateway::{AppState, Gateway, RedisClients, Resources};
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
    matchers::{body_partial_json, method, path},
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

async fn inference_for(config: &Config) -> Arc<Gateway> {
    litellm_gateway::build_inference(config, &Resources::build(config).unwrap())
        .await
        .unwrap()
}

fn global(router: axum::Router) -> axum::Router {
    litellm_gateway::middleware::global(
        router,
        &litellm_gateway::middleware::Settings::default(),
        Arc::new(Entitlements::Community),
        &litellm_gateway::middleware::InFlight::new(),
    )
}

#[fixture]
async fn inference() -> Arc<Gateway> {
    inference_for(&Config::from_yaml("model_list: []").unwrap()).await
}

fn without_stores(inference: Arc<Gateway>) -> AppState {
    AppState {
        inference,
        redis: RedisClients::default(),
        database: None,
        entitlements: Arc::new(Entitlements::Community),
        environment: Arc::new(litellm_env::ProcessEnvironment),
    }
}

#[rstest]
#[case::authorized("/v1/messages", &[("authorization", "Bearer gateway-key")], Some("gateway-key"), 400, None)]
#[case::x_api_key("/v1/messages", &[("x-api-key", "gateway-key")], Some("gateway-key"), 400, None)]
#[case::wrong_x_api_key("/v1/messages", &[("x-api-key", "wrong")], Some("gateway-key"), 401, Some("missing or invalid API key"))]
#[case::missing_token("/v1/messages", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::wrong_token("/v1/messages", &[("authorization", "Bearer wrong")], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::ocr("/ocr", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::chat("/v1/chat/completions", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::deployment(
    "/openai/deployments/model/chat/completions",
    &[],
    Some("gateway-key"),
    401,
    Some("missing or invalid bearer token")
)]
#[case::transcription("/audio/transcriptions", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::unsupported_route("/responses", &[], Some("gateway-key"), 401, Some("missing or invalid bearer token"))]
#[case::unknown_path("/unknown", &[], Some("gateway-key"), 404, None)]
#[case::unknown_path_unconfigured("/unknown", &[], None, 404, None)]
#[case::unconfigured("/v1/messages", &[("authorization", "Bearer gateway-key")], None, 500, None)]
#[tokio::test]
async fn authenticates_before_serving_mounted_inference_routes(
    #[future(awt)] inference: Arc<Gateway>,
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
        .resources
        .pool
        .client(&inference.http, ClientVariant::Provider)
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
async fn logs_request_outcome_without_credentials_or_query(#[future(awt)] inference: Arc<Gateway>) {
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
    #[future(awt)] inference: Arc<Gateway>,
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
    #[future(awt)] inference: Arc<Gateway>,
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

async fn routed(config: &str, model: &str) -> axum::response::Response {
    let config = Config::from_yaml(config).unwrap();
    let request = Request::post("/v1/messages")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();
    global(litellm_gateway::router(
        without_stores(inference_for(&config).await),
        &config,
    ))
    .oneshot(request)
    .await
    .unwrap()
}

fn routing_headers(response: &axum::response::Response) -> Vec<(&str, &str)> {
    response
        .headers()
        .iter()
        .filter(|(name, _)| name.as_str().starts_with("x-litellm-") && *name != "x-litellm-call-id")
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

#[rstest]
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

#[rstest]
#[case::strategy("routing_strategy: usage-based-routing-v2", "usage-based-routing-v2")]
#[case::pre_call_checks("enable_pre_call_checks: true", "enable_pre_call_checks")]
#[case::tag_filtering("enable_tag_filtering: true", "enable_tag_filtering")]
#[case::health_check_routing("enable_health_check_routing: true", "enable_health_check_routing")]
#[case::affinity("model_group_affinity_config: {}", "model_group_affinity_config")]
#[case::default_slots(
    "default_max_parallel_requests: 2",
    "router_settings.default_max_parallel_requests"
)]
#[tokio::test]
async fn a_router_setting_the_gateway_cannot_serve_fails_the_build(
    #[case] setting: &str,
    #[case] named: &str,
) {
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9\nrouter_settings:\n  {setting}\n"
    ))
    .unwrap();

    let refused = litellm_gateway::build_inference(&config, &Resources::build(&config).unwrap())
        .await
        .err()
        .map(|error| litellm_error::Report::new(&error).to_string())
        .unwrap();

    assert!(refused.contains(named), "{refused}");
}

#[tokio::test]
async fn a_deployment_asking_for_slots_fails_the_build() {
    let config = Config::from_yaml(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: anthropic/claude-unlisted-9\n      max_parallel_requests: 2\n",
    )
    .unwrap();

    let refused = litellm_gateway::build_inference(&config, &Resources::build(&config).unwrap())
        .await
        .err()
        .map(|error| error.to_string());

    assert_eq!(
        refused.as_deref(),
        Some("the gateway does not honor `model_list.litellm_params.max_parallel_requests` yet")
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

#[tokio::test]
async fn a_stream_carrying_the_providers_error_cools_its_deployment_down() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(streamed(concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-unlisted-9\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        )))
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

    let first = stream_read_to_its_end(&app, "primary").await;
    let second = stream_read_to_its_end(&app, "primary").await;
    let (status, refused) = stream_read_to_its_end(&app, "primary").await;

    assert_eq!((first.0, second.0), (200, 200));
    assert!(first.1.contains("overloaded_error"), "{}", first.1);
    assert_eq!(
        (status, serde_json::from_str::<Value>(&refused).unwrap()),
        (
            429,
            json!({"type": "error", "error": {
                "type": "rate_limit_error",
                "message": "No deployments available for selected model, Try again in 30 seconds. All deployments for selected model are in cooldown. Passed model=primary. pre-call-checks=False, cooldown_list=['primary-1', 'primary-2']",
            }})
        )
    );
    assert_eq!(upstream.received_requests().await.unwrap().len(), 2);
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
    let config = litellm_gateway::load_config(&path).await.unwrap();
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

#[rstest]
#[case::refused_by_default("", "", 501, 0)]
#[case::sent_when_client_side_credentials_are_allowed(
    "",
    "  allow_client_side_credentials: true\n",
    200,
    1
)]
#[case::sent_when_the_deployment_opts_into_the_endpoint(
    "      configurable_clientside_auth_params: [aws_bedrock_runtime_endpoint]\n",
    "",
    200,
    1
)]
#[case::refused_when_the_deployment_opts_into_another_param(
    "      configurable_clientside_auth_params: [aws_role_name]\n",
    "",
    501,
    0
)]
#[tokio::test]
async fn chat_sends_a_caller_supplied_bedrock_endpoint_only_under_client_side_credentials(
    #[case] deployment_settings: &str,
    #[case] general_settings: &str,
    #[case] status: u16,
    #[case] caller_host_requests: u64,
) {
    let upstream = MockServer::start().await;
    let caller_host = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"inferenceConfig": {"maxTokens": 16}}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
        })))
        .expect(caller_host_requests)
        .mount(&caller_host)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: bedrock/us-east-1/anthropic.claude-v2\n      api_base: {}\n      api_key: test-key\n{deployment_settings}general_settings:\n  master_key: gateway-key\n{general_settings}",
        upstream.uri()
    ))
    .unwrap();
    let body = json!({
        "model": "public-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "aws_bedrock_runtime_endpoint": caller_host.uri()
    });
    let request = Request::post("/v1/chat/completions")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = litellm_gateway::router(without_stores(inference_for(&config).await), &config)
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), status);
}

#[rstest]
#[case::dropped_under_drop_params("      drop_params: true\n", None, None)]
#[case::sent_without_drop_params("", Some(json!(0.5)), Some(json!(0.9)))]
#[tokio::test]
async fn chat_drops_sampling_params_the_catalog_says_the_model_rejects(
    #[case] deployment_settings: &str,
    #[case] temperature: Option<Value>,
    #[case] top_p: Option<Value>,
) {
    let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
    let (model, _) = catalog
        .served()
        .find(|(name, fields)| {
            name.starts_with("anthropic.")
                && fields.get("litellm_provider").and_then(Value::as_str)
                    == Some("bedrock_converse")
                && fields.get("supports_sampling_params") == Some(&Value::Bool(false))
        })
        .unwrap();
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
        })))
        .mount(&upstream)
        .await;
    let config = Config::from_yaml(&format!(
        "model_list:\n  - model_name: public-model\n    litellm_params:\n      model: bedrock/{model}\n      api_base: {}\n      api_key: test-key\n{deployment_settings}general_settings:\n  master_key: gateway-key\n",
        upstream.uri()
    ))
    .unwrap();
    let body = json!({
        "model": "public-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "temperature": 0.5,
        "top_p": 0.9
    });
    let request = Request::post("/v1/chat/completions")
        .header("authorization", "Bearer gateway-key")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = litellm_gateway::router(without_stores(inference_for(&config).await), &config)
        .oneshot(request)
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 200);
    let forwarded: Value =
        serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        forwarded["inferenceConfig"].get("temperature").cloned(),
        temperature
    );
    assert_eq!(forwarded["inferenceConfig"].get("topP").cloned(), top_p);
    assert_eq!(forwarded["inferenceConfig"]["maxTokens"], json!(16));
}
