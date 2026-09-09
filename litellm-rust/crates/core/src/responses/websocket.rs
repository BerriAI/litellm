use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures_util::{Sink, SinkExt, Stream, StreamExt};
use rustls::{ClientConfig, RootCertStore};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config,
};

use crate::Error;
use crate::integrations::custom_logger::CallbackTiming;
use crate::lifecycle::{
    ExecutedCall, TerminalClassification, TerminalDispatcher,
};
use crate::providers::dispatch::responses_websocket_provider_config;
use crate::responses::instrumentation::ResponsesWsInstrumentation;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType, ResponsesWsTransformResult};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

type Upstream = WebSocketStream<MaybeTlsStream<TcpStream>>;
static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

struct ProviderConnection {
    upstream: Upstream,
    config: &'static dyn ResponsesWebSocketProviderConfig,
}

pub trait ResponsesWebSocketProviderConfig: Sync {
    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Err(Error::Unsupported("provider credential resolution"))
    }

    fn supports_native_websocket(&self) -> bool {
        false
    }

    fn model_in_websocket_url(&self) -> bool {
        true
    }

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> String {
        complete_websocket_url(api_base, model, self.model_in_websocket_url())
    }

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;

    fn transform_ws_response(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;
}

pub struct ResponsesWebSocketRequest {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub first_frame: Option<ResponsesWsEvent>,
    pub idle_timeout: Option<Duration>,
}

pub async fn responses_websocket<S, In, Out>(
    services: Arc<S>,
    request: ResponsesWebSocketRequest,
    context: crate::lifecycle::CallLifecycleContext,
    client_in: In,
    client_out: Out,
) -> Result<ExecutedCall<(), Error>, Error>
where
    S: TerminalDispatcher + crate::lifecycle::Clock,
    S: 'static,
    In: Stream<Item = Result<ResponsesWsEvent, Error>> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let start_time = services.now();
    let instrumentation = Arc::new(ResponsesWsInstrumentation::default());
    let snapshot = instrumentation.clone();
    let mut completion = crate::lifecycle::completion::SessionCompletion::new(
        services, context, start_time, move |context| {
            let observation = snapshot.snapshot();
            if !observation.model.is_empty() { context.model = observation.model; }
            context.usage = observation.usage;
        },
    );
    let config = responses_websocket_provider_config();
    let connection = async {
        let key = config.resolve_api_key(request.api_key.as_deref(), &|key| std::env::var(key).ok())?;
        dial_upstream(config, &request.model, &key, request.api_base.as_deref()).await
    }.await;
    let upstream = match connection {
        Ok(upstream) => upstream,
        Err(error) => {
            let terminal = completion.settle(TerminalClassification::Failure {
                kind: "ConnectionError".into(), message: error.to_string(),
            }).await;
            return Ok(ExecutedCall::Failure { error, terminal });
        }
    };
    let result = splice(
        upstream,
        &request.model,
        request.first_frame,
        request.idle_timeout.unwrap_or(IDLE_TIMEOUT),
        instrumentation.as_ref(),
        client_in,
        client_out,
    )
    .await;
    let classification = match &result {
        Ok(classification) => classification.clone(),
        Err(failure) => failure.classification.clone(),
    };
    let terminal = completion.settle(classification).await;
    Ok(match result {
        Ok(TerminalClassification::Success) => ExecutedCall::Success {
            response: (),
            terminal,
        },
        Ok(TerminalClassification::Failure { message, .. } | TerminalClassification::Cancelled { message } | TerminalClassification::Incomplete { message }) => ExecutedCall::Failure {
            error: Error::InvalidResponse(message),
            terminal,
        },
        Err(failure) => ExecutedCall::Failure {
            error: failure.error,
            terminal,
        },
    })
}

struct ResponsesWsFailure {
    error: Error,
    classification: TerminalClassification,
}

impl ResponsesWsFailure {
    fn new(error: Error) -> Self {
        Self {
            classification: TerminalClassification::Failure {
                kind: error_kind(&error).to_string(),
                message: error.to_string(),
            },
            error,
        }
    }

    fn session(kind: &str, message: &str) -> Self {
        Self {
            error: Error::Network(message.to_string()),
            classification: TerminalClassification::Failure {
                kind: kind.to_string(),
                message: message.to_string(),
            },
        }
    }
}

async fn splice<In, Out>(
    connection: ProviderConnection,
    model: &str,
    first_frame: Option<ResponsesWsEvent>,
    idle_timeout: Duration,
    instrumentation: &ResponsesWsInstrumentation,
    mut client_in: In,
    mut client_out: Out,
) -> Result<TerminalClassification, ResponsesWsFailure>
where
    In: Stream<Item = Result<ResponsesWsEvent, Error>> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let ProviderConnection { upstream, config } = connection;
    let (mut upstream_tx, mut upstream_rx) = upstream.split();
    if let Some(event) = first_frame {
        send_provider_event(config, &mut upstream_tx, &event, model)
            .await
            .map_err(ResponsesWsFailure::new)?;
    }
    loop {
        tokio::select! {
            event = client_in.next() => {
                let Some(event) = event else {
                    return Err(ResponsesWsFailure::session(
                        "ClientDisconnected",
                        "client disconnected before response.completed",
                    ));
                };
                let event = event.map_err(ResponsesWsFailure::new)?;
                send_provider_event(config, &mut upstream_tx, &event, model).await.map_err(ResponsesWsFailure::new)?;
            }
            message = upstream_rx.next() => {
                let Some(message) = message else {
                    return Err(ResponsesWsFailure::session(
                        "ProviderDisconnected",
                        "provider disconnected before a terminal response frame",
                    ));
                };
                match message.map_err(ws_transport_error).map_err(ResponsesWsFailure::new)? {
                    Message::Text(text) => {
                        let event = serde_json::from_str::<ResponsesWsEvent>(&text)
                            .map_err(|error| ResponsesWsFailure::new(Error::InvalidResponse(error.to_string())))?;
                        instrumentation.observe(&event);
                        let terminal = instrumentation.terminal_classification(&event);
                        for outbound in config.transform_ws_response(&event, model)
                            .map_err(ResponsesWsFailure::new)?.events {
                            client_out.send(outbound).await
                                .map_err(|error| ResponsesWsFailure::session(
                                    "ClientDisconnected",
                                    &format!("failed to deliver provider event to client: {error}"),
                                ))?;
                        }
                        if let Some(classification) = terminal {
                            return Ok(classification);
                        }
                    }
                    Message::Close(_) => return Err(ResponsesWsFailure::session(
                        "ProviderDisconnected",
                        "provider closed before a terminal response frame",
                    )),
                    _ => {}
                }
            }
            _ = tokio::time::sleep(idle_timeout) => return Err(ResponsesWsFailure::session(
                "IdleTimeout",
                "Responses WebSocket session timed out before a terminal response frame",
            )),
        }
    }
}

async fn send_provider_event(
    config: &dyn ResponsesWebSocketProviderConfig,
    upstream: &mut futures_util::stream::SplitSink<Upstream, Message>,
    event: &ResponsesWsEvent,
    model: &str,
) -> Result<(), Error> {
    for outbound in config.transform_ws_request(event, model)?.events {
        let payload = serde_json::to_string(&outbound)
            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
        upstream
            .send(Message::Text(payload))
            .await
            .map_err(ws_transport_error)?;
    }
    Ok(())
}

async fn dial_upstream(
    config: &'static dyn ResponsesWebSocketProviderConfig,
    model: &str,
    api_key: &str,
    api_base: Option<&str>,
) -> Result<ProviderConnection, Error> {
    let url = config.complete_websocket_url(api_base, model);
    let mut request = url.into_client_request().map_err(ws_transport_error)?;
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|error| Error::Auth(error.to_string()))?,
    );
    let connector = match request.uri().scheme_str() {
        Some("wss") => Some(Connector::Rustls(tls_config()?)),
        _ => None,
    };
    let connect = connect_async_tls_with_config(request, None, false, connector);
    let result = tokio::time::timeout(CONNECT_TIMEOUT, connect)
        .await
        .map_err(|_| Error::Connect("Responses WebSocket connection timed out".to_string()))?;
    result
        .map(|(upstream, _)| ProviderConnection { upstream, config })
        .map_err(ws_handshake_error)
}

fn tls_config() -> Result<Arc<ClientConfig>, Error> {
    if let Some(config) = TLS_CONFIG.get() {
        return Ok(Arc::clone(config));
    }
    let native = rustls_native_certs::load_native_certs();
    let mut roots = RootCertStore::empty();
    let (added, _) = roots.add_parsable_certificates(native.certs);
    if added == 0 {
        return Err(Error::Connect(format!(
            "no usable native root certificates: {:?}",
            native.errors
        )));
    }
    let config =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|error| Error::Connect(error.to_string()))?
            .with_root_certificates(roots)
            .with_no_client_auth();
    let config = Arc::new(config);
    Ok(Arc::clone(TLS_CONFIG.get_or_init(|| config)))
}

fn ws_handshake_error(error: WsError) -> Error {
    match error {
        WsError::Http(response) => Error::Http {
            status: response.status().as_u16(),
            body: response
                .body()
                .as_ref()
                .map(|body| String::from_utf8_lossy(body).into_owned())
                .unwrap_or_default(),
        },
        other => ws_transport_error(other),
    }
}

fn ws_transport_error(error: WsError) -> Error {
    match error {
        WsError::Io(error) => Error::Connect(error.to_string()),
        other => Error::Network(other.to_string()),
    }
}

pub use crate::providers::openai::responses::transformation::complete_websocket_url;

pub fn enforce_model(event: &ResponsesWsEvent, model: &str) -> ResponsesWsEvent {
    if !event.is_response_create() {
        return event.clone();
    }
    let mut enforced = event.clone();
    let has_flat_model = enforced.data.contains_key("model");
    if let Some(response) = enforced
        .data
        .get_mut("response")
        .and_then(Value::as_object_mut)
    {
        response.insert("model".to_string(), Value::String(model.to_string()));
        if has_flat_model {
            enforced
                .data
                .insert("model".to_string(), Value::String(model.to_string()));
        }
    } else {
        enforced
            .data
            .insert("model".to_string(), Value::String(model.to_string()));
    }
    enforced
}

pub fn is_terminal_event(event_type: &ResponsesWsEventType) -> bool {
    matches!(
        event_type,
        ResponsesWsEventType::ResponseCompleted
            | ResponsesWsEventType::ResponseFailed
            | ResponsesWsEventType::ResponseIncomplete
            | ResponsesWsEventType::Error
    )
}

fn error_kind(error: &Error) -> &'static str {
    match error {
        Error::Auth(_) => "AuthError",
        Error::InvalidProvider(_) => "InvalidProvider",
        Error::InvalidRequest(_) => "InvalidRequest",
        Error::InvalidType { .. } => "InvalidType",
        Error::MissingField(_) => "MissingField",
        Error::Http { .. } => "HttpError",
        Error::InvalidResponse(_) => "InvalidResponse",
        Error::Network(_) => "NetworkError",
        Error::Connect(_) => "ConnectError",
        Error::Routing(_) => "RoutingError",
        Error::Unsupported(_) => "UnsupportedRequest",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{TerminalRecord, RouteProjection};
    use crate::integrations::custom_logger::{LogError, LogFuture};
    use crate::lifecycle::{CallLifecycleContext, Clock};
    use std::sync::Mutex;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    #[derive(Default)]
    struct Services {
        terminals: Mutex<Vec<TerminalRecord>>,
    }
    impl Clock for Services {
        fn now(&self) -> f64 {
            1.0
        }
    }
    impl TerminalDispatcher for Services {
        fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
            Box::pin(async move {
                self.terminals.lock().unwrap().push(terminal.clone());
                Ok::<(), LogError>(())
            })
        }
    }

    fn event(value: Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("event")
    }

    async fn provider_with_frames(
        frames: Vec<Value>,
        remain_open: bool,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            for frame in frames {
                socket.send(Message::Text(frame.to_string())).await.unwrap();
            }
            if remain_open {
                futures_util::future::pending::<()>().await;
            } else {
                socket.close(None).await.unwrap();
            }
        });
        (format!("http://{address}"), task)
    }

    type InputChannel = (
        futures_channel::mpsc::UnboundedSender<Result<ResponsesWsEvent, Error>>,
        futures_channel::mpsc::UnboundedReceiver<Result<ResponsesWsEvent, Error>>,
    );

    fn input() -> InputChannel {
        futures_channel::mpsc::unbounded()
    }

    fn assert_failure(services: &Services, kind: &str, message: &str) {
        let terminals = services.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            terminals[0].classification,
            TerminalClassification::Failure {
                kind: kind.to_string(),
                message: message.to_string(),
            }
        );
    }

    #[tokio::test]
    async fn completed_frame_is_delivered_and_settles_once_while_provider_remains_open() {
        let (api_base, server) = provider_with_frames(
            vec![json!({"type":"response.completed","response":{"id":"resp-1","model":"authorized","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}})],
            true,
        )
        .await;
        let services = Arc::new(Services::default());
        let (client_tx, client_rx) = input();
        let (output_tx, mut output_rx) = futures_channel::mpsc::unbounded();
        client_tx
            .unbounded_send(Ok(event(json!({"type":"response.create","model":"wrong"}))))
            .unwrap();
        let result = responses_websocket(
            Arc::clone(&services),
            ResponsesWebSocketRequest {
                model: "authorized".into(),
                api_key: Some("key".into()),
                api_base: Some(api_base),
                first_frame: None,
                idle_timeout: Some(Duration::from_secs(1)),
            },
            CallLifecycleContext::new("responses_websocket", "authorized", "openai", "call-1"),
            client_rx,
            output_tx,
        )
        .await
        .unwrap();
        assert!(matches!(result, ExecutedCall::Success { .. }));
        assert_eq!(
            output_rx.next().await.unwrap().event_type,
            ResponsesWsEventType::ResponseCompleted
        );
        let terminals = services.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert_eq!(terminals[0].classification, TerminalClassification::Success);
        assert_eq!(terminals[0].usage.total_tokens, 3);
        server.abort();
    }

    #[tokio::test]
    async fn provider_failure_terminals_are_failures_and_are_delivered_once() {
        let cases = [
            (
                json!({"type":"response.failed","response":{"error":{"message":"request rejected"}}}),
                "ResponseFailed",
                "request rejected",
            ),
            (
                json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}),
                "ResponseIncomplete",
                "max_output_tokens",
            ),
            (
                json!({"type":"error","error":{"message":"provider unavailable"}}),
                "ProviderError",
                "provider unavailable",
            ),
        ];
        for (frame, kind, message) in cases {
            let expected_type = frame["type"].as_str().unwrap().to_string();
            let (api_base, server) = provider_with_frames(vec![frame], true).await;
            let services = Arc::new(Services::default());
            let (_client_tx, client_rx) = input();
            let (output_tx, mut output_rx) = futures_channel::mpsc::unbounded();
            let result = responses_websocket(
                Arc::clone(&services),
                ResponsesWebSocketRequest {
                    model: "model".into(),
                    api_key: Some("key".into()),
                    api_base: Some(api_base),
                    first_frame: None,
                    idle_timeout: Some(Duration::from_secs(1)),
                },
                CallLifecycleContext::new("responses_websocket", "model", "openai", "call-1"),
                client_rx,
                output_tx,
            )
            .await
            .unwrap();
            assert!(matches!(result, ExecutedCall::Failure { .. }));
            assert_eq!(
                output_rx.next().await.unwrap().event_type.as_ref(),
                expected_type
            );
            assert_failure(services.as_ref(), kind, message);
            server.abort();
        }
    }

    #[tokio::test]
    async fn handshake_status_is_preserved_without_a_terminal() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream
                .write_all(b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\n\r\nslow")
                .await
                .unwrap();
        });
        let services = Arc::new(Services::default());
        let (_, input): (
            _,
            futures_channel::mpsc::UnboundedReceiver<Result<ResponsesWsEvent, Error>>,
        ) = input();
        let (output, _) = futures_channel::mpsc::unbounded();
        let error = responses_websocket(
            Arc::clone(&services),
            ResponsesWebSocketRequest {
                model: "model".into(),
                api_key: Some("key".into()),
                api_base: Some(format!("http://{address}")),
                first_frame: None,
                idle_timeout: None,
            },
            CallLifecycleContext::new("responses_websocket", "model", "openai", "call-1"),
            input,
            output,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Http { status: 429, .. }));
        assert!(services.terminals.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn committed_protocol_failure_dispatches_one_terminal() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            socket.send(Message::Text("not-json".into())).await.unwrap();
        });
        let services = Arc::new(Services::default());
        let (client_tx, input) = input();
        let (output, _) = futures_channel::mpsc::unbounded();
        let result = responses_websocket(
            Arc::clone(&services),
            ResponsesWebSocketRequest {
                model: "model".into(),
                api_key: Some("key".into()),
                api_base: Some(format!("http://{address}")),
                first_frame: None,
                idle_timeout: None,
            },
            CallLifecycleContext::new("responses_websocket", "model", "openai", "call-1"),
            input,
            output,
        )
        .await
        .unwrap();
        drop(client_tx);
        assert!(matches!(
            result,
            ExecutedCall::Failure {
                error: Error::InvalidResponse(_),
                ..
            }
        ));
        let terminals = services.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            terminals[0].classification,
            TerminalClassification::Failure { .. }
        ));
    }

    #[tokio::test]
    async fn client_drop_provider_close_and_idle_timeout_are_distinct_failures() {
        let cases = [
            (
                true,
                true,
                "ClientDisconnected",
                "client disconnected before response.completed",
            ),
            (
                false,
                false,
                "ProviderDisconnected",
                "provider closed before a terminal response frame",
            ),
            (
                false,
                true,
                "IdleTimeout",
                "Responses WebSocket session timed out before a terminal response frame",
            ),
        ];
        for (drop_client, remain_open, kind, message) in cases {
            let (api_base, server) = provider_with_frames(Vec::new(), remain_open).await;
            let services = Arc::new(Services::default());
            let (client_tx, client_rx) = input();
            if drop_client {
                drop(client_tx);
            }
            let (output_tx, _) = futures_channel::mpsc::unbounded();
            let result = responses_websocket(
                Arc::clone(&services),
                ResponsesWebSocketRequest {
                    model: "model".into(),
                    api_key: Some("key".into()),
                    api_base: Some(api_base),
                    first_frame: None,
                    idle_timeout: Some(Duration::from_millis(20)),
                },
                CallLifecycleContext::new("responses_websocket", "model", "openai", "call-1"),
                client_rx,
                output_tx,
            )
            .await
            .unwrap();
            assert!(matches!(result, ExecutedCall::Failure { .. }));
            assert_failure(services.as_ref(), kind, message);
            server.abort();
        }
    }

    #[tokio::test]
    async fn cancelling_a_committed_session_dispatches_one_failure() {
        let (api_base, server) = provider_with_frames(Vec::new(), true).await;
        let services = Arc::new(Services::default());
        let (_client_tx, client_rx) = input();
        let (output_tx, _) = futures_channel::mpsc::unbounded();
        let task = tokio::spawn(responses_websocket(
            Arc::clone(&services),
            ResponsesWebSocketRequest {
                model: "model".into(),
                api_key: Some("key".into()),
                api_base: Some(api_base),
                first_frame: None,
                idle_timeout: Some(Duration::from_secs(60)),
            },
            CallLifecycleContext::new("responses_websocket", "model", "openai", "call-1"),
            client_rx,
            output_tx,
        ));
        tokio::time::sleep(Duration::from_millis(20)).await;
        task.abort();
        let _ = task.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            while services.terminals.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_failure(
            services.as_ref(),
            "Cancelled",
            "Responses WebSocket session was cancelled before completion",
        );
        server.abort();
    }

    #[tokio::test]
    async fn client_protocol_error_dispatches_one_failure() {
        let (api_base, server) = provider_with_frames(Vec::new(), true).await;
        let services = Arc::new(Services::default());
        let (client_tx, client_rx) = input();
        client_tx
            .unbounded_send(Err(Error::InvalidRequest(
                "invalid client frame".to_string(),
            )))
            .unwrap();
        let (output_tx, _) = futures_channel::mpsc::unbounded();
        let result = responses_websocket(
            Arc::clone(&services),
            ResponsesWebSocketRequest {
                model: "model".into(),
                api_key: Some("key".into()),
                api_base: Some(api_base),
                first_frame: None,
                idle_timeout: Some(Duration::from_secs(1)),
            },
            CallLifecycleContext::new("responses_websocket", "model", "openai", "call-1"),
            client_rx,
            output_tx,
        )
        .await
        .unwrap();
        assert!(matches!(
            result,
            ExecutedCall::Failure {
                error: Error::InvalidRequest(_),
                ..
            }
        ));
        assert_failure(
            services.as_ref(),
            "InvalidRequest",
            "invalid request: invalid client frame",
        );
        server.abort();
    }

    #[test]
    fn url_and_model_behavior_match_the_public_protocol() {
        assert_eq!(
            complete_websocket_url(None, "gpt-5", true),
            "wss://api.openai.com/v1/responses?model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("http://localhost:8080/"), "gpt 5", true),
            "ws://localhost:8080/responses?model=gpt%205"
        );
        let nested = enforce_model(
            &event(json!({"type":"response.create","model":"wrong","response":{"model":"wrong"}})),
            "right",
        );
        assert_eq!(nested.model(), Some("right"));
        assert_eq!(nested.data["response"]["model"], "right");
    }
}
