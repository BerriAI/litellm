use std::{
    fs,
    io::Write,
    net::SocketAddr,
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::{Request, State},
    http::{
        HeaderMap, HeaderValue, Method, StatusCode, Uri,
        header::{AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE},
    },
    response::Response,
};
use futures_util::{SinkExt, StreamExt, stream};
use http_body_util::BodyExt;
use litellm_gateway_inference::MAX_BODY_BYTES;
use litellm_http::Client;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tempfile::NamedTempFile;
use tokio::{
    io::AsyncReadExt,
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::{sleep, timeout},
};
use tokio_tungstenite::{accept_async, client_async, tungstenite::Message};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
static PROCESS_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct GatewayProcess {
    child: Child,
    address: SocketAddr,
    logs: NamedTempFile,
    _config: Option<NamedTempFile>,
    _lock: tokio::sync::MutexGuard<'static, ()>,
}

impl GatewayProcess {
    async fn spawn(
        flag: Option<&str>,
        upstream: Option<&str>,
        extra: &[(&str, &str)],
        config: Option<&str>,
    ) -> Self {
        let lock = PROCESS_LOCK.lock().await;
        let reserved = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = reserved.local_addr().unwrap();
        drop(reserved);

        let config = config.map(|contents| {
            let mut file = NamedTempFile::new().unwrap();
            file.write_all(contents.as_bytes()).unwrap();
            file
        });
        let logs = NamedTempFile::new().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_litellm-gateway"));
        command
            .env_clear()
            .env("HOST", address.ip().to_string())
            .env("PORT", address.port().to_string())
            .env("RUST_LOG", "debug")
            .envs(extra.iter().copied())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(logs.as_file().try_clone().unwrap()));
        if let Some(flag) = flag {
            command.env("LITELLM_RUST_GATEWAY", flag);
        }
        if let Some(upstream) = upstream {
            command.env("LITELLM_PYTHON_UPSTREAM", upstream);
        }
        if let Some(config) = config.as_ref() {
            command.env("LITELLM_CONFIG", config.path());
        }
        let child = command.spawn().unwrap();

        Self {
            child,
            address,
            logs,
            _config: config,
            _lock: lock,
        }
    }

    async fn wait_until_ready(&mut self) {
        timeout(STARTUP_TIMEOUT, async {
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    panic!(
                        "gateway exited before listening with {status}: {}",
                        self.read_logs()
                    );
                }
                if TcpStream::connect(self.address).await.is_ok() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("gateway should listen within the startup timeout");
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }

    async fn wait_for_exit(mut self) -> String {
        let status = timeout(STARTUP_TIMEOUT, async {
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    break status;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("invalid startup should exit within the timeout");
        assert!(
            !status.success(),
            "invalid startup should exit unsuccessfully"
        );
        self.read_logs()
    }

    fn stop(mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.read_logs()
    }

    fn read_logs(&self) -> String {
        fs::read_to_string(self.logs.path()).unwrap()
    }
}

impl Drop for GatewayProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct ServerHandle {
    address: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl ServerHandle {
    async fn start(app: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        Self {
            address,
            shutdown: Some(shutdown),
            task,
        }
    }

    fn origin(&self) -> String {
        format!("http://{}", self.address)
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.abort();
    }
}

#[derive(Debug)]
struct ObservedRequest {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
}

struct RecordingUpstream {
    server: ServerHandle,
    received: mpsc::UnboundedReceiver<ObservedRequest>,
}

impl RecordingUpstream {
    async fn echo() -> Self {
        let (sender, received) = mpsc::unbounded_channel();
        let app = Router::new().fallback(echo_request).with_state(sender);
        Self {
            server: ServerHandle::start(app).await,
            received,
        }
    }

    async fn delegated_auth() -> Self {
        let (sender, received) = mpsc::unbounded_channel();
        let app = Router::new()
            .fallback(delegated_auth_request)
            .with_state(sender);
        Self {
            server: ServerHandle::start(app).await,
            received,
        }
    }

    fn origin(&self) -> String {
        self.server.origin()
    }

    async fn next(&mut self) -> ObservedRequest {
        timeout(REQUEST_TIMEOUT, self.received.recv())
            .await
            .expect("upstream should receive the request")
            .expect("upstream request channel should remain open")
    }
}

#[fixture]
async fn echo_upstream() -> RecordingUpstream {
    RecordingUpstream::echo().await
}

async fn read_request(
    sender: &mpsc::UnboundedSender<ObservedRequest>,
    request: Request,
) -> (HeaderMap, Bytes) {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, MAX_BODY_BYTES + 1024 * 1024).await.unwrap();
    sender
        .send(ObservedRequest {
            method: parts.method,
            uri: parts.uri,
            headers: parts.headers.clone(),
            body: body.clone(),
        })
        .unwrap();
    (parts.headers, body)
}

fn copy_header(headers: &HeaderMap, response: &mut Response, name: &'static str) {
    if let Some(value) = headers.get(name) {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_bytes(value.as_bytes()).unwrap());
    }
}

async fn echo_request(
    State(sender): State<mpsc::UnboundedSender<ObservedRequest>>,
    request: Request,
) -> Response {
    let (headers, body) = read_request(&sender, request).await;
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::ACCEPTED;
    response.headers_mut().insert(
        "x-upstream-secret",
        HeaderValue::from_static("response-secret"),
    );
    copy_header(&headers, &mut response, CONTENT_TYPE.as_str());
    copy_header(&headers, &mut response, "mcp-session-id");
    copy_header(&headers, &mut response, "last-event-id");
    copy_header(&headers, &mut response, "mcp-protocol-version");
    response
}

async fn delegated_auth_request(
    State(sender): State<mpsc::UnboundedSender<ObservedRequest>>,
    request: Request,
) -> Response {
    let (headers, _) = read_request(&sender, request).await;
    let allowed = headers
        .get(AUTHORIZATION)
        .is_some_and(|value| value == "Bearer delegated-allow");
    let (status, body, decision) = if allowed {
        (StatusCode::CREATED, "python-only model response", "allowed")
    } else {
        (StatusCode::FORBIDDEN, "delegated auth denied", "rejected")
    };
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("x-auth-decision", HeaderValue::from_static(decision));
    if !allowed {
        response.headers_mut().insert(
            WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=python"),
        );
    }
    response
}

async fn forward_gateway(upstream: &str, extra: &[(&str, &str)]) -> GatewayProcess {
    let mut gateway = GatewayProcess::spawn(Some("true"), Some(upstream), extra, None).await;
    gateway.wait_until_ready().await;
    gateway
}

fn client() -> Client {
    Client::for_test()
}

#[rstest]
#[case::chat_completions("/v1/chat/completions", "PATCH")]
#[case::messages("/v1/messages", "PUT")]
#[case::responses("/v1/responses", "DELETE")]
#[case::liveliness("/health/liveliness", "POST")]
#[case::readiness("/health/readiness", "PATCH")]
#[case::health("/health", "OPTIONS")]
#[case::ui("/ui", "POST")]
#[case::mcp("/mcp", "PUT")]
#[tokio::test]
async fn forwards_every_path_method_and_body_to_python(
    #[future] echo_upstream: RecordingUpstream,
    #[case] path: &str,
    #[case] method: &str,
) {
    let mut upstream = echo_upstream.await;
    let gateway = forward_gateway(&upstream.origin(), &[]).await;
    let body = format!("arbitrary body for {path}");
    let method = Method::from_bytes(method.as_bytes()).unwrap();

    let response = client()
        .request(method.clone(), gateway.url(path))
        .header("x-client-value", "preserved")
        .body(body.clone())
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(response.bytes().await.unwrap(), body.as_bytes());
    let observed = upstream.next().await;
    assert_eq!(observed.method, method);
    assert_eq!(observed.uri.path(), path);
    assert_eq!(observed.headers["x-client-value"], "preserved");
    assert_eq!(
        observed.headers["x-litellm-gateway-declined"],
        "route not native"
    );
    assert_eq!(observed.body, body.as_bytes());
}

#[rstest]
#[case::missing_config(None, &[("LITELLM_CONFIG", "/missing/gateway-config.yaml")])]
#[case::invalid_config_and_ui(
    Some("model_list: ["),
    &[
        ("LITELLM_UI_PATH", "/missing/dashboard"),
        ("LITELLM_UI_SECURE_COOKIES", "not-a-bool"),
    ],
)]
#[tokio::test]
async fn forward_only_starts_before_config_and_ui_loading(
    #[future] echo_upstream: RecordingUpstream,
    #[case] config: Option<&str>,
    #[case] extra: &[(&str, &str)],
) {
    let mut upstream = echo_upstream.await;
    let mut gateway =
        GatewayProcess::spawn(Some("true"), Some(&upstream.origin()), extra, config).await;
    gateway.wait_until_ready().await;

    let response = client()
        .post(gateway.url("/v1/chat/completions"))
        .body("python owns this request")
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(response.bytes().await.unwrap(), "python owns this request");
    assert_eq!(upstream.next().await.uri.path(), "/v1/chat/completions");
}

#[rstest]
#[case::allowed("Bearer delegated-allow", StatusCode::CREATED)]
#[case::rejected("Bearer delegated-deny", StatusCode::FORBIDDEN)]
#[tokio::test]
async fn python_only_models_and_delegated_auth_match_direct_python(
    #[case] authorization: &str,
    #[case] expected: StatusCode,
) {
    let mut upstream = RecordingUpstream::delegated_auth().await;
    let gateway = forward_gateway(&upstream.origin(), &[]).await;
    let body = json!({"model": "python-only", "messages": [{"role": "user", "content": "hello"}]})
        .to_string();
    let direct = client()
        .post(format!("{}/v1/chat/completions", upstream.origin()))
        .header(AUTHORIZATION, authorization)
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();
    let direct_status = direct.status();
    let direct_decision = direct.headers()["x-auth-decision"].clone();
    let direct_challenge = direct.headers().get(WWW_AUTHENTICATE).cloned();
    let direct_body = direct.bytes().await.unwrap();
    let forwarded = client()
        .post(gateway.url("/v1/chat/completions"))
        .header(AUTHORIZATION, authorization)
        .header(CONTENT_TYPE, "application/json")
        .body(body.clone())
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(direct_status, expected);
    assert_eq!(forwarded.status(), direct_status);
    assert_eq!(forwarded.headers()["x-auth-decision"], direct_decision);
    assert_eq!(
        forwarded.headers().get(WWW_AUTHENTICATE),
        direct_challenge.as_ref()
    );
    assert_eq!(forwarded.bytes().await.unwrap(), direct_body);
    let _direct_request = upstream.next().await;
    let forwarded_request = upstream.next().await;
    assert_eq!(forwarded_request.headers[AUTHORIZATION], authorization);
    assert_eq!(forwarded_request.body, body.as_bytes());
}

#[rstest]
#[tokio::test]
async fn opaque_and_mcp_transport_data_round_trip_unchanged(
    #[future] echo_upstream: RecordingUpstream,
) {
    let mut upstream = echo_upstream.await;
    let gateway = forward_gateway(&upstream.origin(), &[]).await;
    let body = Bytes::from_static(b"v=0\r\n\0\xffbinary");

    let response = client()
        .post(gateway.url("/mcp?transport=opaque"))
        .header(CONTENT_TYPE, "application/sdp")
        .header("mcp-session-id", "session-123")
        .header("last-event-id", "event-456")
        .header("mcp-protocol-version", "2025-06-18")
        .body(body.clone())
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(response.headers()[CONTENT_TYPE], "application/sdp");
    assert_eq!(response.headers()["mcp-session-id"], "session-123");
    assert_eq!(response.headers()["last-event-id"], "event-456");
    assert_eq!(response.headers()["mcp-protocol-version"], "2025-06-18");
    assert_eq!(response.bytes().await.unwrap(), body);
    let observed = upstream.next().await;
    assert_eq!(
        observed.uri.path_and_query().unwrap(),
        "/mcp?transport=opaque"
    );
    assert_eq!(observed.headers[CONTENT_TYPE], "application/sdp");
    assert_eq!(observed.headers["mcp-session-id"], "session-123");
    assert_eq!(observed.headers["last-event-id"], "event-456");
    assert_eq!(observed.headers["mcp-protocol-version"], "2025-06-18");
    assert_eq!(observed.body, body);
}

struct StreamingState {
    first_chunk: tokio::sync::Mutex<Option<oneshot::Sender<()>>>,
}

async fn count_streamed_body(
    State(state): State<Arc<StreamingState>>,
    request: Request,
) -> Response {
    let mut body = request.into_body();
    let mut size = 0;
    while let Some(frame) = body.frame().await {
        let frame = frame.unwrap();
        if let Some(data) = frame.data_ref() {
            size += data.len();
            if let Some(first_chunk) = state.first_chunk.lock().await.take() {
                let _ = first_chunk.send(());
            }
        }
    }
    Response::new(Body::from(size.to_string()))
}

#[rstest]
#[tokio::test]
async fn streams_uploads_larger_than_the_native_body_limit_before_eof() {
    let (first_chunk, first_chunk_received) = oneshot::channel();
    let upstream = ServerHandle::start(Router::new().fallback(count_streamed_body).with_state(
        Arc::new(StreamingState {
            first_chunk: tokio::sync::Mutex::new(Some(first_chunk)),
        }),
    ))
    .await;
    let gateway = forward_gateway(&upstream.origin(), &[]).await;
    let (release, released) = oneshot::channel::<()>();
    let first =
        stream::once(async { Ok::<_, std::io::Error>(Bytes::from_static(b"streamed-before-eof")) });
    let remainder = stream::once(async move {
        let _ = released.await;
        Ok(Bytes::from(vec![b'x'; MAX_BODY_BYTES]))
    });
    let request = tokio::spawn(
        client()
            .post(gateway.url("/large-python-upload"))
            .body(reqwest::Body::wrap_stream(first.chain(remainder)))
            .timeout(Duration::from_secs(30))
            .send(),
    );

    timeout(REQUEST_TIMEOUT, first_chunk_received)
        .await
        .expect("upstream should receive the first chunk before request EOF")
        .unwrap();
    release.send(()).unwrap();
    let response = timeout(Duration::from_secs(30), request)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let received = response.text().await.unwrap().parse::<usize>().unwrap();
    assert!(received > MAX_BODY_BYTES);
}

struct PartialDisconnect {
    address: SocketAddr,
    body_seen: Option<oneshot::Receiver<()>>,
    task: Option<JoinHandle<usize>>,
}

impl PartialDisconnect {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (body_seen, seen) = oneshot::channel();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut received = Vec::new();
            let mut chunk = [0; 4096];
            loop {
                let size = socket.read(&mut chunk).await.unwrap();
                if size == 0 {
                    break;
                }
                received.extend_from_slice(&chunk[..size]);
                let body_started = received
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .is_some_and(|headers| received.len() > headers + 4);
                if body_started {
                    break;
                }
            }
            let _ = body_seen.send(());
            drop(socket);
            match timeout(Duration::from_secs(2), listener.accept()).await {
                Ok(Ok((socket, _))) => {
                    drop(socket);
                    2
                }
                _ => 1,
            }
        });
        Self {
            address,
            body_seen: Some(seen),
            task: Some(task),
        }
    }

    fn origin(&self) -> String {
        format!("http://{}", self.address)
    }

    async fn body_seen(&mut self) {
        timeout(
            REQUEST_TIMEOUT,
            self.body_seen.take().expect("body receiver should exist"),
        )
        .await
        .expect("upstream should consume part of the request")
        .unwrap();
    }

    async fn finish(mut self) -> usize {
        timeout(REQUEST_TIMEOUT, self.task.take().unwrap())
            .await
            .unwrap()
            .unwrap()
    }
}

impl Drop for PartialDisconnect {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[rstest]
#[tokio::test]
async fn does_not_retry_after_upstream_consumes_a_partial_body_and_disconnects() {
    let mut upstream = PartialDisconnect::start().await;
    let gateway = forward_gateway(&upstream.origin(), &[]).await;
    let (release, released) = oneshot::channel::<()>();
    let first =
        stream::once(async { Ok::<_, std::io::Error>(Bytes::from_static(b"first request bytes")) });
    let remainder = stream::once(async move {
        let _ = released.await;
        Ok(Bytes::from_static(b"remaining request bytes"))
    });
    let request = tokio::spawn(
        client()
            .post(gateway.url("/partial-disconnect"))
            .body(reqwest::Body::wrap_stream(first.chain(remainder)))
            .timeout(REQUEST_TIMEOUT)
            .send(),
    );

    upstream.body_seen().await;
    release.send(()).unwrap();
    let response = timeout(REQUEST_TIMEOUT, request)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(upstream.finish().await, 1);
}

#[rstest]
#[tokio::test]
async fn unreachable_upstream_returns_bad_gateway() {
    let unreachable = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", unreachable.local_addr().unwrap());
    let gateway = forward_gateway(&origin, &[]).await;
    drop(unreachable);

    let response = client()
        .post(gateway.url("/python-only"))
        .body("request")
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[rstest]
#[case::invalid_flag(
    Some("gateway-flag-secret"),
    Some("http://127.0.0.1:4002"),
    "gateway-flag-secret"
)]
#[case::missing_upstream(Some("true"), None, "configuration-secret")]
#[case::empty_upstream(Some("true"), Some(""), "configuration-secret")]
#[case::https(
    Some("true"),
    Some("https://upstream-secret.invalid"),
    "upstream-secret"
)]
#[case::path(
    Some("true"),
    Some("http://127.0.0.1/upstream-secret"),
    "upstream-secret"
)]
#[case::query(
    Some("true"),
    Some("http://127.0.0.1?token=upstream-secret"),
    "upstream-secret"
)]
#[case::fragment(
    Some("true"),
    Some("http://127.0.0.1/#upstream-secret"),
    "upstream-secret"
)]
#[case::userinfo(
    Some("true"),
    Some("http://upstream-secret@127.0.0.1"),
    "upstream-secret"
)]
#[case::malformed(Some("true"), Some("http://[upstream-secret"), "upstream-secret")]
#[case::malformed_port(
    Some("true"),
    Some("http://127.0.0.1:upstream-secret"),
    "upstream-secret"
)]
#[case::zero_port(Some("true"), Some("http://127.0.0.1:0"), ":0")]
#[case::overflow_port(Some("true"), Some("http://127.0.0.1:65536"), "65536")]
#[tokio::test]
async fn invalid_startup_inputs_fail_closed_without_echoing_values(
    #[case] flag: Option<&str>,
    #[case] upstream: Option<&str>,
    #[case] forbidden: &str,
) {
    let process = GatewayProcess::spawn(flag, upstream, &[], None).await;

    let logs = process.wait_for_exit().await;

    assert!(!logs.contains(forbidden));
}

const NATIVE_CONFIG: &str = "model_list: []\ngeneral_settings:\n  master_key: gateway-key\n";

#[rstest]
#[case::unset(None)]
#[case::explicit_false(Some("false"))]
#[tokio::test]
async fn disabled_forward_only_flag_keeps_native_auth_health_and_dispatch(
    #[case] flag: Option<&str>,
) {
    let mut gateway = GatewayProcess::spawn(flag, None, &[], Some(NATIVE_CONFIG)).await;
    gateway.wait_until_ready().await;

    let auth = client()
        .post(gateway.url("/v1/chat/completions"))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"python-only","messages":[]}"#)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();
    let health = client()
        .get(gateway.url("/health"))
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(auth.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(health.status(), StatusCode::NOT_FOUND);
}

#[rstest]
#[tokio::test]
async fn python_upstream_without_enable_flag_keeps_native_gateway(
    #[future] echo_upstream: RecordingUpstream,
) {
    let mut upstream = echo_upstream.await;
    let mut gateway =
        GatewayProcess::spawn(None, Some(&upstream.origin()), &[], Some(NATIVE_CONFIG)).await;
    gateway.wait_until_ready().await;

    let native = client()
        .post(gateway.url("/v1/chat/completions"))
        .header(CONTENT_TYPE, "application/json")
        .body(r#"{"model":"python-only","messages":[]}"#)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(native.status(), StatusCode::UNAUTHORIZED);
    assert!(upstream.received.try_recv().is_err());

    let fallback = client()
        .post(gateway.url("/python-only"))
        .body("fallback request")
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();

    assert_eq!(fallback.status(), StatusCode::ACCEPTED);
    assert_eq!(upstream.next().await.uri.path(), "/python-only");
}

#[rstest]
#[tokio::test]
async fn executable_forward_only_mode_inherits_websocket_lifecycle() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let upstream = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        let message = socket.next().await.unwrap().unwrap();
        socket.send(message).await.unwrap();
    });
    let gateway = forward_gateway(&origin, &[]).await;
    let stream = timeout(REQUEST_TIMEOUT, TcpStream::connect(gateway.address))
        .await
        .unwrap()
        .unwrap();
    let (mut socket, response) = timeout(
        REQUEST_TIMEOUT,
        client_async(
            format!("ws://{}/realtime?model=python-only", gateway.address),
            stream,
        ),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    socket.send(Message::text("through gateway")).await.unwrap();
    assert_eq!(
        timeout(REQUEST_TIMEOUT, socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        Message::text("through gateway")
    );
    drop(socket);
    timeout(REQUEST_TIMEOUT, upstream).await.unwrap().unwrap();
}

fn outcome_record(logs: &str) -> Value {
    logs.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|record| record["message"] == "response headers")
        .expect("forward-only request should emit one outcome record")
}

#[rstest]
#[tokio::test]
async fn debug_logs_only_request_outcome_metadata(#[future] echo_upstream: RecordingUpstream) {
    let upstream = echo_upstream.await;
    let origin = upstream.origin();
    let gateway = forward_gateway(&origin, &[]).await;

    let response = client()
        .post(gateway.url("/v1/messages?token=query-secret"))
        .header(AUTHORIZATION, "Bearer header-secret")
        .body("request-body-secret")
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(response.headers()["x-upstream-secret"], "response-secret");
    assert_eq!(response.bytes().await.unwrap(), "request-body-secret");
    let logs = gateway.stop();
    let record = outcome_record(&logs);

    assert_eq!(record["fields"]["status"], 202);
    assert_eq!(record["fields"]["method"], "POST");
    assert_eq!(record["fields"]["path"], "/v1/messages");
    assert!(record["fields"]["request_id"].as_str().is_some());
    assert!(!logs.contains("header-secret"));
    assert!(!logs.contains("query-secret"));
    assert!(!logs.contains("request-body-secret"));
    assert!(!logs.contains("response-secret"));
    assert!(!logs.contains(&origin));
}
