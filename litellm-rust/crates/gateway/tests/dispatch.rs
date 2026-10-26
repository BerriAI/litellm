use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::{ConnectInfo, DefaultBodyLimit},
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, stream};
use litellm_config::Config;
use litellm_gateway::{Fallback, Forward};
use litellm_gateway_inference::MAX_BODY_BYTES;
use rstest::rstest;
use serde_json::{Value, json};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
    time::timeout,
};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

fn app(python: &str, model: &str, provider: &str) -> Router {
    let config = Config::from_yaml(&format!(
        "general_settings:\n  master_key: gateway-key\nmodel_list:\n  - model_name: public/model\n    litellm_params:\n      model: {model}\n      api_base: {provider}\n      api_key: test-key\n"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let fallback = Fallback::Forward(Forward::new(python).unwrap());
    litellm_gateway::router(inference, &config, None, None, fallback)
}

async fn send(app: Router, path: &str, body: Value) -> Response {
    app.oneshot(
        Request::post(path)
            .header("content-type", "application/json")
            .header("authorization", "Bearer gateway-key")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn python() -> MockServer {
    let python = MockServer::start().await;
    Mock::given(any())
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-served-by", "python")
                .set_body_string("from python"),
        )
        .mount(&python)
        .await;
    python
}

async fn provider() -> MockServer {
    let provider = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(500).set_body_string("provider failed"))
        .mount(&provider)
        .await;
    provider
}

#[rstest]
#[case::chat_streaming(
    "anthropic/test-model",
    "/v1/chat/completions?trace=1",
    r#"{"model": "public/model",  "messages": [{"role": "user", "content": "hi"}], "stream": true}"#,
    "streaming"
)]
#[case::messages_bedrock_invoke(
    "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "/v1/messages",
    r#"{"model":"public/model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
    "Bedrock invoke messages request shaping"
)]
#[case::responses_provider(
    "gemini/test-model",
    "/v1/responses",
    r#"{"model":"public/model","input":"hi"}"#,
    "native HTTP responses provider"
)]
#[tokio::test]
async fn a_core_decline_reaches_python_as_the_client_sent_it(
    #[case] model: &str,
    #[case] path: &str,
    #[case] body: &'static str,
    #[case] reason: &str,
) {
    let python = python().await;
    let provider = provider().await;
    let app = app(&python.uri(), model, &provider.uri());

    let response = app
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer gateway-key")
                .header("x-client-header", "kept")
                .header("connection", "te")
                .header("te", "trailers")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-served-by"], "python");
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        "from python"
    );
    assert!(provider.received_requests().await.unwrap().is_empty());
    let forwarded = python.received_requests().await.unwrap();
    assert_eq!(forwarded.len(), 1);
    let forwarded = &forwarded[0];
    assert_eq!(forwarded.method.as_str(), "POST");
    let forwarded_path = match forwarded.url.query() {
        Some(query) => format!("{}?{query}", forwarded.url.path()),
        None => forwarded.url.path().to_owned(),
    };
    assert_eq!(forwarded_path, path);
    assert_eq!(forwarded.body, body.as_bytes());
    assert_eq!(forwarded.headers["authorization"], "Bearer gateway-key");
    assert_eq!(forwarded.headers["x-client-header"], "kept");
    assert_eq!(forwarded.headers["x-litellm-gateway-declined"], reason);
    assert!(!forwarded.headers.contains_key("te"));
}

#[rstest]
#[case::invalid_request(json!({"model": "public/model", "messages": "hi"}), 400)]
#[case::unknown_model(json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}), 400)]
#[case::provider_failure(json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}), 500)]
#[tokio::test]
async fn a_terminal_error_is_never_forwarded(#[case] body: Value, #[case] status: u16) {
    let python = python().await;
    let provider = provider().await;
    let response = send(
        app(&python.uri(), "anthropic/test-model", &provider.uri()),
        "/v1/chat/completions",
        body,
    )
    .await;

    assert_eq!(response.status(), status);
    assert!(python.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::direct_client(None, "203.0.113.7")]
#[case::behind_a_load_balancer(Some("198.51.100.1"), "198.51.100.1, 203.0.113.7")]
#[tokio::test]
async fn a_forwarded_decline_names_the_client_in_x_forwarded_for(
    #[case] incoming: Option<&str>,
    #[case] expected: &str,
) {
    let python = python().await;
    let provider = provider().await;
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer gateway-key");
    let request = match incoming {
        Some(chain) => request.header("x-forwarded-for", chain),
        None => request,
    };
    let mut request = request
        .body(Body::from(
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "stream": true})
                .to_string(),
        ))
        .unwrap();
    request.extensions_mut().insert(ConnectInfo(
        "203.0.113.7:50000".parse::<SocketAddr>().unwrap(),
    ));

    app(&python.uri(), "anthropic/test-model", &provider.uri())
        .oneshot(request)
        .await
        .unwrap();

    let forwarded = python.received_requests().await.unwrap();
    assert_eq!(forwarded[0].headers["x-forwarded-for"], expected);
}

async fn listen(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(litellm_gateway::serve(
        listener,
        app,
        std::future::pending(),
    ));
    address
}

async fn gateway_in_front_of(python: SocketAddr) -> SocketAddr {
    listen(app(
        &format!("http://{python}"),
        "anthropic/test-model",
        "http://127.0.0.1:1",
    ))
    .await
}

async fn recording_python(
    respond: fn() -> Response,
) -> (SocketAddr, mpsc::UnboundedReceiver<(HeaderMap, Bytes)>) {
    let (sender, received) = mpsc::unbounded_channel();
    let python = Router::new()
        .fallback(move |headers: HeaderMap, body: Bytes| async move {
            sender.send((headers, body)).unwrap();
            respond()
        })
        .layer(DefaultBodyLimit::disable());
    (listen(python).await, received)
}

fn streaming_chat(content: &str) -> String {
    json!({"model": "public/model", "stream": true, "messages": [{"role": "user", "content": content}]})
        .to_string()
}

fn streaming_chat_of_size(size: usize) -> String {
    streaming_chat(&"a".repeat(size - streaming_chat("").len()))
}

fn chat(gateway: SocketAddr, body: String) -> reqwest::RequestBuilder {
    litellm_http::Client::for_test()
        .post(format!("http://{gateway}/v1/chat/completions"))
        .header("content-type", "application/json")
        .header("authorization", "Bearer gateway-key")
        .timeout(Duration::from_secs(10))
        .body(body)
}

#[rstest]
#[tokio::test]
async fn a_forwarded_stream_reaches_the_client_before_python_finishes_it() {
    let (finish, finished) = oneshot::channel::<()>();
    let finished = Arc::new(Mutex::new(Some(finished)));
    let python = listen(Router::new().fallback(move || async move {
        let finished = finished.lock().unwrap().take().unwrap();
        let first = stream::once(async { Ok::<_, Infallible>(Bytes::from("data: first\n\n")) });
        let last = stream::once(async move {
            let _ = finished.await;
            Ok(Bytes::from("data: last\n\n"))
        });
        (
            [("content-type", "text/event-stream")],
            Body::from_stream(first.chain(last)),
        )
    }))
    .await;
    let gateway = gateway_in_front_of(python).await;

    let response = chat(gateway, streaming_chat("hi")).send().await.unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut body = response.bytes_stream();
    let first = timeout(Duration::from_secs(5), body.next())
        .await
        .expect("the first chunk arrives while Python holds the stream open");
    assert_eq!(first.unwrap().unwrap(), "data: first\n\n");
    finish.send(()).unwrap();
    let rest: Vec<Bytes> = body.map(Result::unwrap).collect().await;
    assert_eq!(rest.concat(), b"data: last\n\n");
}

#[rstest]
#[tokio::test]
async fn python_answers_the_client_with_its_own_status_and_end_to_end_headers() {
    let (python, _received) = recording_python(|| {
        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                ("retry-after", "7"),
                ("connection", "x-python-hop"),
                ("x-python-hop", "internal"),
            ],
            "slow down",
        )
            .into_response()
    })
    .await;
    let gateway = gateway_in_front_of(python).await;
    let direct = litellm_http::Client::for_test()
        .post(format!("http://{python}/"))
        .send()
        .await
        .unwrap();
    assert_eq!(direct.headers()["x-python-hop"], "internal");

    let response = chat(gateway, streaming_chat("hi")).send().await.unwrap();

    assert_eq!(response.status(), 429);
    assert_eq!(response.headers()["retry-after"], "7");
    assert!(!response.headers().contains_key("x-python-hop"));
    assert_eq!(response.text().await.unwrap(), "slow down");
}

#[rstest]
#[tokio::test]
async fn python_sees_the_connecting_client_at_the_end_of_x_forwarded_for() {
    let (python, mut received) = recording_python(|| "from python".into_response()).await;
    let gateway = gateway_in_front_of(python).await;

    chat(gateway, streaming_chat("hi"))
        .header("x-forwarded-for", "198.51.100.1")
        .send()
        .await
        .unwrap();

    let (headers, _) = received.try_recv().unwrap();
    assert_eq!(headers["x-forwarded-for"], "198.51.100.1, 127.0.0.1");
    assert_eq!(headers["x-litellm-gateway-declined"], "streaming");
}

#[rstest]
#[case::at_the_limit(MAX_BODY_BYTES, 200, true)]
#[case::one_byte_over(MAX_BODY_BYTES + 1, 413, false)]
#[tokio::test]
async fn a_declined_body_up_to_the_limit_reaches_python_byte_for_byte(
    #[case] size: usize,
    #[case] status: u16,
    #[case] forwarded: bool,
) {
    let (python, mut received) = recording_python(|| "from python".into_response()).await;
    let gateway = gateway_in_front_of(python).await;
    let body = streaming_chat_of_size(size);

    let response = chat(gateway, body.clone()).send().await.unwrap();

    assert_eq!(response.status().as_u16(), status);
    let received = received.try_recv().ok().map(|(_, received)| received);
    assert_eq!(received.as_deref() == Some(body.as_bytes()), forwarded);
    assert_eq!(received.is_some(), forwarded);
}

#[rstest]
#[tokio::test]
async fn an_unreachable_python_is_a_bad_gateway() {
    let closed = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let gateway = gateway_in_front_of(closed).await;

    let response = chat(gateway, streaming_chat("hi")).send().await.unwrap();

    assert_eq!(response.status(), 502);
}
