use std::{
    net::SocketAddr,
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use futures_util::StreamExt;
use litellm_config::Config;
use litellm_gateway::{Fallback, Forward};
use litellm_gateway_inference::{Error, Gateway};
use litellm_http::ClientVariant;
use litellm_tracing::{Logger, Metadata, Record, Sink};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
    time::timeout,
};
use tokio_tungstenite::{accept_async, client_async, tungstenite::handshake::derive_accept_key};
use tower::ServiceExt;

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

#[fixture]
fn inference() -> Arc<Gateway> {
    litellm_gateway::build_inference(&Config::from_yaml("model_list: []").unwrap()).unwrap()
}

async fn managed_gateway(
    inference: Arc<Gateway>,
    upstream: SocketAddr,
) -> (
    SocketAddr,
    oneshot::Sender<()>,
    JoinHandle<std::io::Result<()>>,
) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let fallback = Fallback::Forward(Forward::new(&format!("http://{upstream}")).unwrap());
    let app = litellm_gateway::router(inference, &config, None, None, fallback);
    let (shutdown, stopped) = oneshot::channel();
    let server = tokio::spawn(litellm_gateway::serve(listener, app, async move {
        let _ = stopped.await;
    }));
    (address, shutdown, server)
}

fn websocket_key(request: &[u8]) -> String {
    String::from_utf8_lossy(request)
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("sec-websocket-key")
                .then(|| value.trim().to_owned())
        })
        .unwrap()
}

#[rstest]
#[case::authorized("/v1/messages", Some("Bearer gateway-key"), Some("gateway-key"), 400)]
#[case::missing_token("/v1/messages", None, Some("gateway-key"), 401)]
#[case::wrong_token("/v1/messages", Some("Bearer wrong"), Some("gateway-key"), 401)]
#[case::ocr("/ocr", None, Some("gateway-key"), 401)]
#[case::chat("/v1/chat/completions", None, Some("gateway-key"), 401)]
#[case::deployment(
    "/openai/deployments/model/chat/completions",
    None,
    Some("gateway-key"),
    401
)]
#[case::transcription("/audio/transcriptions", None, Some("gateway-key"), 401)]
#[case::unsupported_route("/responses", None, Some("gateway-key"), 401)]
#[case::unknown_path("/unknown", None, Some("gateway-key"), 404)]
#[case::unknown_path_unconfigured("/unknown", None, None, 404)]
#[case::unconfigured("/v1/messages", Some("Bearer gateway-key"), None, 500)]
#[tokio::test]
async fn authenticates_before_serving_mounted_inference_routes(
    inference: Arc<Gateway>,
    #[case] path: &str,
    #[case] authorization: Option<&str>,
    #[case] master_key: Option<&str>,
    #[case] status: u16,
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
            litellm_gateway::router(inference, &config, None, None, Fallback::Reject),
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
    let request = match authorization {
        Some(value) => request.header("authorization", value),
        None => request,
    };
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
async fn logs_request_outcome_without_credentials_or_query(inference: Arc<Gateway>) {
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
            litellm_gateway::router(inference, &config, None, None, Fallback::Reject)
                .oneshot(request),
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
#[tokio::test]
async fn mounts_ui_without_exposing_credentials_or_authorizing_inference(inference: Arc<Gateway>) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: inference-secret\n")
            .unwrap();
    let assets = tempfile::TempDir::new().unwrap();
    std::fs::write(assets.path().join("index.html"), "dashboard").unwrap();
    let backend = litellm_gateway_auth::UiBackend::new(
        "admin".into(),
        litellm_auth_types::SecretValue::new("ui-password"),
    )
    .unwrap();
    let ui = litellm_gateway_ui::router(
        backend,
        tower_sessions_moka_store::MokaStore::new(Some(10_000)),
        false,
    )
    .merge(litellm_gateway_ui::dashboard_assets(assets.path()));
    let app = litellm_gateway::router(inference, &config, Some(ui), None, Fallback::Reject);
    let page = app
        .clone()
        .oneshot(Request::get("/ui/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(page.status().as_u16(), 200);
    assert_eq!(
        axum::body::to_bytes(page.into_body(), 65536).await.unwrap(),
        "dashboard"
    );
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(LogSink(sender));
    let response = logger
        .instrument(
            app.clone().oneshot(
                Request::post("/v2/login")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"username": "admin", "password": "ui-password"}).to_string(),
                    ))
                    .unwrap(),
            ),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(
        response
            .headers()
            .get_all("set-cookie")
            .iter()
            .all(|cookie| !cookie.to_str().unwrap().contains("Secure"))
    );
    let cookie = response
        .headers()
        .get_all("set-cookie")
        .iter()
        .filter_map(|value| value.to_str().unwrap().split(';').next())
        .collect::<Vec<_>>()
        .join("; ");
    let login: Value = serde_json::from_slice(
        &logger
            .instrument(axum::body::to_bytes(response.into_body(), 65536))
            .await
            .unwrap(),
    )
    .unwrap();
    let jwt = login["token"].as_str().unwrap();
    let token: Value = serde_json::from_slice(
        &URL_SAFE_NO_PAD
            .decode(jwt.split('.').nth(1).unwrap())
            .unwrap(),
    )
    .unwrap();
    let csrf = token["key"].as_str().unwrap();
    let logs = receiver.try_iter().collect::<Vec<_>>();
    let logged = serde_json::to_string(&logs).unwrap();
    assert!(!logged.contains("ui-password"));
    assert!(!logged.contains(jwt));
    let ui_response = app
        .clone()
        .oneshot(
            Request::get("/session/info")
                .header("cookie", &cookie)
                .header("authorization", format!("Bearer {csrf}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ui_response.status().as_u16(), 200);
    let inference_response = app
        .clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("cookie", cookie)
                .header("authorization", format!("Bearer {csrf}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(inference_response.status().as_u16(), 401);
    let master_response = app
        .oneshot(
            Request::get("/session/info")
                .header("authorization", "Bearer inference-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(master_response.status().as_u16(), 401);
}

#[rstest]
#[case::assets("/ui/", "GET")]
#[case::login("/v2/login", "POST")]
#[case::session("/session/info", "GET")]
#[case::discovery("/.well-known/litellm-ui-config", "GET")]
#[tokio::test]
async fn ui_routes_are_absent_when_not_mounted(
    inference: Arc<Gateway>,
    #[case] path: &str,
    #[case] method: &str,
) {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let response = litellm_gateway::router(inference, &config, None, None, Fallback::Reject)
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 404);
}

#[rstest]
#[case::unserved_path("POST", "/v1/embeddings", None, true)]
#[case::management_api("GET", "/key/info?key=sk-1", None, true)]
#[case::wrong_method_on_a_native_path("GET", "/v1/messages", None, true)]
#[case::bad_key_on_a_native_route("POST", "/v1/messages", Some("Bearer wrong"), false)]
#[tokio::test]
async fn python_serves_every_request_the_gateway_does_not_route(
    inference: Arc<Gateway>,
    #[case] method: &str,
    #[case] path: &str,
    #[case] authorization: Option<&str>,
    #[case] forwarded: bool,
) {
    let python = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::any())
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("from python"))
        .mount(&python)
        .await;
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let fallback = Fallback::Forward(Forward::new(&python.uri()).unwrap());
    let request = Request::builder().method(method).uri(path);
    let request = match authorization {
        Some(value) => request.header("authorization", value),
        None => request,
    };

    let mut request = request.body(Body::from("{}")).unwrap();
    request.extensions_mut().insert(axum::extract::ConnectInfo(
        "203.0.113.7:50000".parse::<std::net::SocketAddr>().unwrap(),
    ));

    let response = litellm_gateway::router(inference, &config, None, None, fallback)
        .oneshot(request)
        .await
        .unwrap();

    let received = python.received_requests().await.unwrap();
    assert_eq!(received.len(), usize::from(forwarded));
    if forwarded {
        assert_eq!(response.status().as_u16(), 200);
        assert_eq!(received[0].method.as_str(), method);
        let url = &received[0].url;
        let forwarded_path = match url.query() {
            Some(query) => format!("{}?{query}", url.path()),
            None => url.path().to_owned(),
        };
        assert_eq!(forwarded_path, path);
        assert_eq!(
            received[0].headers["x-litellm-gateway-declined"],
            "route not native"
        );
        assert_eq!(received[0].headers["x-forwarded-for"], "203.0.113.7");
    } else {
        assert_eq!(response.status().as_u16(), 401);
    }
}

#[rstest]
#[tokio::test]
async fn serve_drains_an_open_websocket_and_closes_both_peers(inference: Arc<Gateway>) {
    let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = upstream_listener.local_addr().unwrap();
    let (ready_sender, ready_receiver) = oneshot::channel();
    let (eof_sender, eof_receiver) = oneshot::channel();
    let upstream_task = tokio::spawn(async move {
        let (stream, _) = upstream_listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        ready_sender.send(()).unwrap();
        let result = timeout(Duration::from_secs(7), socket.next())
            .await
            .unwrap();
        eof_sender
            .send(result.is_none_or(|message| message.is_err()))
            .unwrap();
    });

    let (gateway, shutdown, server) = managed_gateway(inference, upstream).await;
    let stream = TcpStream::connect(gateway).await.unwrap();
    let (mut socket, response) = timeout(
        Duration::from_secs(2),
        client_async(format!("ws://{gateway}/realtime"), stream),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    timeout(Duration::from_secs(2), ready_receiver)
        .await
        .unwrap()
        .unwrap();

    let started = Instant::now();
    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(6), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(started.elapsed() <= Duration::from_secs(6));

    let result = timeout(Duration::from_secs(2), socket.next())
        .await
        .unwrap();
    assert!(result.is_none_or(|message| message.is_err()));
    assert!(
        timeout(Duration::from_secs(2), eof_receiver)
            .await
            .unwrap()
            .unwrap()
    );
    upstream_task.await.unwrap();
}

#[rstest]
#[tokio::test]
async fn shutdown_deadline_closes_a_handshake_released_after_shutdown(inference: Arc<Gateway>) {
    let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = upstream_listener.local_addr().unwrap();
    let (request_sender, request_receiver) = oneshot::channel();
    let (release_sender, release_receiver) = oneshot::channel();
    let (closed_sender, closed_receiver) = oneshot::channel();
    let upstream_task = tokio::spawn(async move {
        let (mut stream, _) = upstream_listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut chunk = [0; 4096];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut chunk).await.unwrap();
            if read == 0 {
                return;
            }
            request.extend_from_slice(&chunk[..read]);
        }
        let key = websocket_key(&request);
        request_sender.send(()).unwrap();
        release_receiver.await.unwrap();
        let accept = derive_accept_key(key.as_bytes());
        let response = format!(
            "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        let mut bytes = [0; 64];
        let closed = matches!(
            timeout(Duration::from_secs(3), stream.read(&mut bytes)).await,
            Ok(Ok(0))
        );
        closed_sender.send(closed).unwrap();
    });

    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway = listener.local_addr().unwrap();
    let fallback = Fallback::Forward(Forward::new(&format!("http://{upstream}")).unwrap());
    let app = litellm_gateway::router(inference, &config, None, None, fallback);
    let (shutdown, stopped) = oneshot::channel();
    let (began_sender, began_receiver) = oneshot::channel();
    let server = tokio::spawn(litellm_gateway::serve(listener, app, async move {
        let _ = stopped.await;
        began_sender.send(()).unwrap();
    }));

    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(gateway).await.unwrap();
        let request = format!(
            "GET /realtime HTTP/1.1\r\nHost: {gateway}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        let mut chunk = [0; 1024];
        while !response.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut chunk).await.unwrap();
            if read == 0 {
                break;
            }
            response.extend_from_slice(&chunk[..read]);
        }
        (stream, response)
    });
    timeout(Duration::from_secs(2), request_receiver)
        .await
        .unwrap()
        .unwrap();
    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(2), began_receiver)
        .await
        .unwrap()
        .unwrap();
    tokio::time::sleep(Duration::from_secs(4)).await;
    release_sender.send(()).unwrap();

    let (mut socket, response) = timeout(Duration::from_secs(3), client)
        .await
        .unwrap()
        .unwrap();
    assert!(response.starts_with(b"HTTP/1.1 101"));
    let result = timeout(Duration::from_secs(3), socket.read(&mut [0; 64]))
        .await
        .unwrap();
    assert_eq!(result.unwrap(), 0);
    timeout(Duration::from_secs(3), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        timeout(Duration::from_secs(2), closed_receiver)
            .await
            .unwrap()
            .unwrap()
    );
    upstream_task.await.unwrap();
}

#[rstest]
#[tokio::test]
async fn dropping_serve_cancels_an_open_websocket(inference: Arc<Gateway>) {
    let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = upstream_listener.local_addr().unwrap();
    let (ready_sender, ready_receiver) = oneshot::channel();
    let (eof_sender, eof_receiver) = oneshot::channel();
    let upstream_task = tokio::spawn(async move {
        let (stream, _) = upstream_listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        ready_sender.send(()).unwrap();
        let result = timeout(Duration::from_secs(2), socket.next())
            .await
            .unwrap();
        eof_sender
            .send(result.is_none_or(|message| message.is_err()))
            .unwrap();
    });

    let (gateway, _shutdown, server) = managed_gateway(inference, upstream).await;
    let stream = TcpStream::connect(gateway).await.unwrap();
    let (mut socket, response) = timeout(
        Duration::from_secs(2),
        client_async(format!("ws://{gateway}/realtime"), stream),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    timeout(Duration::from_secs(2), ready_receiver)
        .await
        .unwrap()
        .unwrap();

    server.abort();
    assert!(
        timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap_err()
            .is_cancelled()
    );
    let result = timeout(Duration::from_secs(2), socket.next())
        .await
        .unwrap();
    assert!(result.is_none_or(|message| message.is_err()));
    assert!(
        timeout(Duration::from_secs(2), eof_receiver)
            .await
            .unwrap()
            .unwrap()
    );
    upstream_task.await.unwrap();
}

#[rstest]
#[tokio::test]
async fn unmanaged_axum_server_rejects_websocket_forwarding(inference: Arc<Gateway>) {
    let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_address = upstream.local_addr().unwrap();
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: gateway-key\n")
            .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gateway = listener.local_addr().unwrap();
    let fallback = Fallback::Forward(Forward::new(&format!("http://{upstream_address}")).unwrap());
    let app = litellm_gateway::router(inference, &config, None, None, fallback);
    let (shutdown, stopped) = oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = stopped.await;
            })
            .await
    });

    let stream = TcpStream::connect(gateway).await.unwrap();
    let request = axum::http::Request::builder()
        .uri(format!("ws://{gateway}/realtime"))
        .header("Host", gateway.to_string())
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
        .header("Sec-WebSocket-Version", "13")
        .body(())
        .unwrap();
    let error = timeout(Duration::from_secs(2), client_async(request, stream))
        .await
        .unwrap()
        .unwrap_err();
    let response = match error {
        tokio_tungstenite::tungstenite::Error::Http(response) => response,
        error => panic!("expected an HTTP rejection, got {error}"),
    };
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_ne!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    let body = String::from_utf8(response.body().clone().unwrap()).unwrap();
    assert!(body.contains("requires litellm_gateway::serve"));

    shutdown.send(()).unwrap();
    timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
