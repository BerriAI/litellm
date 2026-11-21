mod keys;

use std::{fmt, io::Write, net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderMap, Method, StatusCode, header},
    response::Response,
};
use flate2::{Compression, write::GzEncoder};
use http_body_util::BodyExt;
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use litellm_config::{Config, Environment};
use litellm_gateway::host::auth::AuthStore;
use litellm_gateway::middleware;
use litellm_gateway_forward::ANSWERED_BY_HEADER;
use litellm_license::Entitlements;
use rstest::rstest;
use tokio::{net::TcpListener, sync::mpsc};
use tower::Layer;

const CALL_ID: &str = "x-litellm-call-id";

struct Received {
    path: String,
    headers: HeaderMap,
    body: Bytes,
}

struct Refused;

impl fmt::Display for Refused {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("refused by the test")
    }
}

impl From<Refused> for &'static str {
    fn from(_: Refused) -> Self {
        "test"
    }
}

async fn python() -> (SocketAddr, mpsc::UnboundedReceiver<Received>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let app = Router::new().fallback(move |request: Request| {
        let sender = sender.clone();
        async move {
            let (parts, body) = request.into_parts();
            let body = body.collect().await.unwrap().to_bytes();
            sender
                .send(Received {
                    path: parts.uri.path().to_owned(),
                    headers: parts.headers,
                    body,
                })
                .unwrap();
            "python"
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, receiver)
}

/// The native gateway under `root_path` with its global middleware, in front of `python`,
/// admitting every request or refusing every one.
async fn gateway(python: SocketAddr, admits: bool, root_path: &str) -> SocketAddr {
    let config = Config::from_yaml("model_list: []").unwrap();
    let native = middleware::root_path::layer(middleware::root_path::Settings::from_environment(
        &Environment {
            server_root_path: root_path.into(),
            ..Environment::default()
        },
    ))
    .layer(middleware::global(
        litellm_gateway::router(
            litellm_gateway::AppState {
                inference: litellm_gateway::build_inference(
                    &config,
                    &litellm_gateway::Resources::build(&config).unwrap(),
                    litellm_gateway_inference::RoutingCache::new(
                        None,
                        litellm_gateway_inference::RouterTiers::default(),
                    ),
                )
                .await
                .unwrap(),
                redis: litellm_gateway::RedisClients::default(),
                database: None,
                keys: None,
                entitlements: Arc::new(Entitlements::Community),
                environment: Arc::new(litellm_env::ProcessEnvironment),
            },
            &config,
        ),
        &middleware::Settings::default(),
        Arc::new(Entitlements::Community),
        &middleware::InFlight::new(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(litellm_gateway_forward::serve(
        listener,
        format!("http://{python}").parse().unwrap(),
        Duration::from_secs(5),
        Duration::from_secs(1),
        native,
        move |request: Request| async move {
            if admits {
                Ok(request)
            } else {
                Err((Box::new(request), Refused))
            }
        },
        std::future::pending(),
    ));
    address
}

fn gzip(body: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(body).unwrap();
    encoder.finish().unwrap()
}

async fn send(gateway: SocketAddr, method: Method, path: &str, gzipped: &[u8]) -> Response {
    let request = Request::builder()
        .method(method)
        .uri(format!("http://{gateway}{path}"))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_ENCODING, "gzip")
        .body(Body::from(gzipped.to_vec()))
        .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        Client::builder(TokioExecutor::new())
            .build_http()
            .request(request),
    )
    .await
    .unwrap()
    .unwrap();
    response.map(Body::new)
}

#[rstest]
#[case::no_native_route(Method::POST, "", "/only/python")]
#[case::wrong_method_on_a_native_route(Method::PUT, "", "/v1/messages")]
#[case::native_route(Method::POST, "", "/v1/messages")]
#[case::native_route_under_a_root_path(Method::POST, "/api", "/api/v1/messages")]
#[tokio::test]
async fn python_answers_a_refused_request_as_the_client_sent_it(
    #[case] method: Method,
    #[case] root_path: &str,
    #[case] path: &str,
) {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, false, root_path).await;
    let gzipped = gzip(br#"{"model":"m"}"#);

    let response = send(gateway, method, path, &gzipped).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[ANSWERED_BY_HEADER],
        "python; reason=test"
    );
    assert!(!response.headers().contains_key(CALL_ID));
    assert!(!response.headers().contains_key(header::X_FRAME_OPTIONS));
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        "python"
    );
    let python = received.recv().await.unwrap();
    assert_eq!(python.path, path);
    assert_eq!(python.headers[header::CONTENT_ENCODING], "gzip");
    assert_eq!(python.body, gzipped);
}

#[rstest]
#[case::at_the_root("", "/v1/messages")]
#[case::under_a_root_path("/api", "/api/v1/messages")]
#[tokio::test]
async fn an_admitted_request_gets_the_native_layers_and_never_reaches_python(
    #[case] root_path: &str,
    #[case] path: &str,
) {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, true, root_path).await;

    let response = send(gateway, Method::POST, path, &gzip(br#"{"model":"m"}"#)).await;

    assert_eq!(response.headers()[ANSWERED_BY_HEADER], "rust");
    assert!(response.headers().contains_key(CALL_ID));
    assert_eq!(response.headers()[header::X_FRAME_OPTIONS], "DENY");
    assert!(received.try_recv().is_err());
}

#[rstest]
#[tokio::test]
async fn an_admitted_request_gets_the_native_404_when_no_route_takes_it() {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, true, "").await;

    let response = send(
        gateway,
        Method::POST,
        "/only/python",
        &gzip(br#"{"model":"m"}"#),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(response.headers()[ANSWERED_BY_HEADER], "rust");
    assert!(received.try_recv().is_err());
}

const MASTER_KEY: &str = "Bearer gateway-key";

/// The native gateway built from `config` the way `main` builds it in front of `python`, with the
/// real `Admit` and no variable in its environment.
async fn configured_gateway(python: SocketAddr, config: &str) -> SocketAddr {
    holding_keys(python, config, None).await
}

/// [`configured_gateway`] whose database holds `keys`.
async fn holding_keys(python: SocketAddr, config: &str, keys: Option<AuthStore>) -> SocketAddr {
    litellm_gateway::install_crypto_provider();
    let config = Config::from_yaml(config).unwrap();
    let resources =
        litellm_gateway::Resources::over(&config, Arc::new(|_: &str| None::<String>)).unwrap();
    let state = litellm_gateway::AppState::over(&config, &Environment::default(), &resources)
        .await
        .unwrap();
    let state = litellm_gateway::AppState {
        keys: keys.map(Arc::new).or(state.keys.clone()),
        ..state
    };
    let admit = Arc::new(litellm_gateway::Admit::new(
        &config,
        &Environment::default(),
        &state,
    ));
    let native = middleware::global(
        litellm_gateway::router(state, &config),
        &middleware::Settings::default(),
        Arc::new(Entitlements::Community),
        &middleware::InFlight::new(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(litellm_gateway_forward::serve(
        listener,
        format!("http://{python}").parse().unwrap(),
        Duration::from_secs(5),
        Duration::from_secs(1),
        native,
        move |request: Request| {
            let admit = admit.clone();
            async move {
                admit
                    .admit(request)
                    .await
                    .map_err(|(request, reason)| (request, reason.why))
            }
        },
        std::future::pending(),
    ));
    address
}

const CLAUDE: &str =
    r#"{"model":"claude","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;

async fn post_as(gateway: SocketAddr, path: &str, authorization: &str) -> Response {
    post(gateway, path, authorization, CLAUDE).await
}

async fn post(
    gateway: SocketAddr,
    path: &str,
    authorization: &str,
    body: &'static str,
) -> Response {
    let request = Request::post(format!("http://{gateway}{path}"))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, authorization)
        .body(Body::from(body))
        .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        Client::builder(TokioExecutor::new())
            .build_http()
            .request(request),
    )
    .await
    .unwrap()
    .unwrap();
    response.map(Body::new)
}

async fn provider() -> wiremock::MockServer {
    let provider = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": "claude-haiku-5-5", "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
            })),
        )
        .mount(&provider)
        .await;
    provider
}

/// `claude` on the provider, and `gpt` on a provider the native route is not compared on.
fn serving(provider: &wiremock::MockServer, settings: &str) -> String {
    format!(
        "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_base: {base}\n      api_key: provider-key\n    model_info:\n      id: claude-a\n  - model_name: gpt\n    litellm_params:\n      model: openai/gpt-5.5\n      api_base: {base}\n      api_key: provider-key\n    model_info:\n      id: gpt-a\ngeneral_settings:\n  master_key: gateway-key\n{settings}",
        base = provider.uri()
    )
}

#[rstest]
#[tokio::test]
async fn a_master_key_callers_messages_request_is_answered_natively() {
    let (python, mut received) = python().await;
    let provider = provider().await;
    let gateway = configured_gateway(python, &serving(&provider, "")).await;

    let response = post_as(gateway, "/v1/messages", MASTER_KEY).await;

    assert_eq!(
        (
            response.status(),
            response.headers()[ANSWERED_BY_HEADER].to_str().unwrap(),
            response.headers()["x-litellm-model-id"].to_str().unwrap(),
        ),
        (StatusCode::OK, "rust", "claude-a")
    );
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["model"], "claude");
    assert!(received.try_recv().is_err());
}

const REACHING_ONLY_GPT: &str = "router_settings:\n  model_group_affinity_config:\n    gpt: [deployment_affinity]\n  routing_groups:\n    - group_name: pool\n      models: [gpt]\n      routing_strategy: least-busy\n";

/// The native router is built from the settings as written, and asks its host for no step on a
/// call a host-only setting does not reach.
#[rstest]
#[tokio::test]
async fn a_group_no_host_only_router_setting_reaches_is_answered_natively() {
    let (python, mut received) = python().await;
    let provider = provider().await;
    let gateway = configured_gateway(python, &serving(&provider, REACHING_ONLY_GPT)).await;

    let response = post_as(gateway, "/v1/messages", MASTER_KEY).await;

    assert_eq!(
        (
            response.status(),
            response.headers()[ANSWERED_BY_HEADER].to_str().unwrap(),
        ),
        (StatusCode::OK, "rust")
    );
    assert!(received.try_recv().is_err());
}

const WITH_A_PROXY_PARAM: &str = r#"{"model":"claude","max_tokens":8,"user":"end-user","messages":[{"role":"user","content":"hi"}]}"#;
const GPT: &str = r#"{"model":"gpt","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
const NOPE: &str = r#"{"model":"nope","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;

#[rstest]
#[case::another_route("", "/v1/chat/completions", MASTER_KEY, CLAUDE, "python; reason=route")]
#[case::another_caller(
    "",
    "/v1/messages",
    "Bearer sk-virtual",
    CLAUDE,
    "python; reason=credential"
)]
#[case::a_setting_only_python_honors(
    "litellm_settings:\n  callbacks: [prometheus]\n",
    "/v1/messages",
    MASTER_KEY,
    CLAUDE,
    "python; reason=settings"
)]
#[case::a_param_only_python_reads(
    "",
    "/v1/messages",
    MASTER_KEY,
    WITH_A_PROXY_PARAM,
    "python; reason=body"
)]
#[case::a_group_of_a_provider_off_the_list(
    "",
    "/v1/messages",
    MASTER_KEY,
    GPT,
    "python; reason=model"
)]
#[case::a_model_the_config_does_not_name(
    "",
    "/v1/messages",
    MASTER_KEY,
    NOPE,
    "python; reason=model"
)]
#[tokio::test]
async fn python_answers_what_the_native_app_would_not_answer_the_same(
    #[case] settings: &str,
    #[case] path: &str,
    #[case] authorization: &str,
    #[case] body: &'static str,
    #[case] answered_by: &str,
) {
    let (python, mut received) = python().await;
    let provider = provider().await;
    let gateway = configured_gateway(python, &serving(&provider, settings)).await;

    let response = post(gateway, path, authorization, body).await;

    assert_eq!(response.headers()[ANSWERED_BY_HEADER], answered_by);
    let received = received.recv().await.unwrap();
    assert_eq!(
        (received.path.as_str(), &received.body[..]),
        (path, body.as_bytes())
    );
    assert!(provider.received_requests().await.unwrap().is_empty());
}

const PROVIDER_STREAM: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-haiku-5-5\",\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

/// The milestone's config: two weighted deployments of `claude`, and `claude-down`, which
/// points at a closed port and falls back to `claude`. The provider is handed back so it
/// outlives the gateway that calls it.
async fn milestone_gateway(python: SocketAddr) -> (SocketAddr, wiremock::MockServer) {
    let provider = provider().await;
    wiremock::Mock::given(wiremock::matchers::body_partial_json(
        serde_json::json!({"stream": true}),
    ))
    .respond_with(
        wiremock::ResponseTemplate::new(200).set_body_raw(PROVIDER_STREAM, "text/event-stream"),
    )
    .with_priority(1)
    .mount(&provider)
    .await;
    let closed = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let deployment = |name: &str, id: &str, base: &str, rpm: &str| {
        format!(
            "  - model_name: {name}\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n      api_base: {base}\n{rpm}    model_info:\n      id: {id}\n"
        )
    };
    let config = format!(
        "model_list:\n{}{}{}router_settings:\n  routing_strategy: simple-shuffle\n  num_retries: 1\n  fallbacks:\n    - claude-down: [claude]\ngeneral_settings:\n  master_key: gateway-key\n",
        deployment("claude", "claude-a", &provider.uri(), "      rpm: 300\n"),
        deployment("claude", "claude-b", &provider.uri(), "      rpm: 100\n"),
        deployment(
            "claude-down",
            "claude-down",
            &format!("http://{closed}"),
            ""
        ),
    );
    (configured_gateway(python, &config).await, provider)
}

/// One milestone call: its status, who answered, the routing headers and its body.
async fn milestone_call(
    gateway: SocketAddr,
    model: &str,
    stream: bool,
) -> (u16, String, Option<String>, Option<String>, String) {
    let request = Request::post(format!("http://{gateway}/v1/messages"))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, MASTER_KEY)
        .body(Body::from(
            serde_json::json!({"model": model, "max_tokens": 16, "stream": stream,
                "messages": [{"role": "user", "content": "hi"}]})
            .to_string(),
        ))
        .unwrap();
    let response = Client::builder(TokioExecutor::new())
        .build_http()
        .request(request)
        .await
        .unwrap();
    let text = |name: &str| {
        response
            .headers()
            .get(name)
            .map(|value| value.to_str().unwrap().to_owned())
    };
    let (status, answered_by, group, fallbacks) = (
        response.status().as_u16(),
        text(ANSWERED_BY_HEADER).unwrap(),
        text("x-litellm-model-group"),
        text("x-litellm-attempted-fallbacks"),
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        answered_by,
        group,
        fallbacks,
        String::from_utf8(body.to_vec()).unwrap(),
    )
}

#[rstest]
#[case::shuffle("claude", false, 200, Some(("claude", "0")), r#""model":"claude""#)]
#[case::streamed_shuffle("claude", true, 200, Some(("claude", "0")), r#""model":"claude""#)]
#[case::fallback(
    "claude-down",
    false,
    200,
    Some(("claude", "1")),
    r#""model":"claude-haiku-5-5""#
)]
#[case::streamed_fallback(
    "claude-down",
    true,
    200,
    Some(("claude", "1")),
    r#""model":"claude-down""#
)]
#[tokio::test]
async fn the_milestone_config_is_answered_natively(
    #[case] model: &str,
    #[case] stream: bool,
    #[case] status: u16,
    #[case] routed: Option<(&str, &str)>,
    #[case] reports: &str,
) {
    let (python, mut received) = python().await;
    let (gateway, _provider) = milestone_gateway(python).await;

    let (answered, answered_by, group, fallbacks, body) =
        milestone_call(gateway, model, stream).await;

    assert_eq!(
        (
            answered,
            answered_by.as_str(),
            group.as_deref().zip(fallbacks.as_deref())
        ),
        (status, "rust", routed)
    );
    assert!(body.contains(reports), "{body}");
    assert!(received.try_recv().is_err());
}

#[rstest]
#[tokio::test]
async fn the_milestone_shuffle_follows_the_deployments_weights() {
    let (python, _received) = python().await;
    let (gateway, _provider) = milestone_gateway(python).await;
    let client = Client::builder(TokioExecutor::new()).build_http();
    let served_by = |_: usize| {
        let request = Request::post(format!("http://{gateway}/v1/messages"))
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::AUTHORIZATION, MASTER_KEY)
            .body(Body::from(
                r#"{"model":"claude","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let client = client.clone();
        async move {
            let response = client.request(request).await.unwrap();
            response.headers()["x-litellm-model-id"]
                .to_str()
                .unwrap()
                .to_owned()
        }
    };

    let served: Vec<String> = futures_util::future::join_all((0..400).map(served_by)).await;

    let by_the_heavier = served.iter().filter(|id| *id == "claude-a").count();
    assert!(
        (250..=350).contains(&by_the_heavier),
        "claude-a holds 3 of the 4 parts of the weight and answered {by_the_heavier} of 400"
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn an_admitted_virtual_key_is_looked_up_once() {
    let (python, mut received) = python().await;
    let provider = provider().await;
    let held = keys::Held::holding("sk-virtual", serde_json::json!({})).await;
    let keys = AuthStore::new(keys::memory_cache(), held.database());
    let gateway = holding_keys(python, &serving(&provider, ""), Some(keys)).await;

    let response = post_as(gateway, "/v1/messages", "Bearer sk-virtual").await;

    assert_eq!(
        (
            response.status(),
            response.headers()[ANSWERED_BY_HEADER].to_str().unwrap()
        ),
        (StatusCode::OK, "rust")
    );
    assert_eq!(held.key_reads(), 1);
    assert!(received.try_recv().is_err());
}
