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
use crate::constants::{OPENAI_RESPONSES_DEFAULT_API_BASE, OPENAI_RESPONSES_PATH};
use crate::integrations::custom_logger::CallbackTiming;
use crate::lifecycle::{
    CostInputs, ExecutedCall, RouteProjection, TerminalClassification, TerminalDispatcher,
    TerminalRecord,
};
use crate::providers::openai::responses::transformation::OPENAI_RESPONSES_WS_CONFIG;
use crate::responses::instrumentation::ResponsesWsInstrumentation;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType, ResponsesWsTransformResult};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const MISSING_KEY_MESSAGE: &str = "Missing OpenAI API Key - a Responses WebSocket call is being made but no key was passed via params or the OPENAI_API_KEY environment variable";

type Upstream = WebSocketStream<MaybeTlsStream<TcpStream>>;
static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

pub trait ResponsesWebSocketProviderConfig: Sync {
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
    services: &S,
    request: ResponsesWebSocketRequest,
    context: crate::lifecycle::CallLifecycleContext,
    client_in: In,
    client_out: Out,
) -> Result<ExecutedCall<(), Error>, Error>
where
    S: TerminalDispatcher + crate::lifecycle::Clock,
    In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let key = resolve_api_key(request.api_key.as_deref())?;
    let upstream = dial_upstream(&request.model, &key, request.api_base.as_deref()).await?;
    let start_time = services.now();
    let instrumentation = ResponsesWsInstrumentation::default();
    let result = splice(
        upstream,
        &request.model,
        request.first_frame,
        request.idle_timeout.unwrap_or(IDLE_TIMEOUT),
        &instrumentation,
        client_in,
        client_out,
    )
    .await;
    let observation = instrumentation.snapshot();
    let model = if observation.model.is_empty() {
        context.model.clone()
    } else {
        observation.model
    };
    let classification = match &result {
        Ok(()) => TerminalClassification::Success,
        Err(error) => TerminalClassification::Failure {
            kind: error_kind(error).to_string(),
            message: error.to_string(),
        },
    };
    let projection = match &classification {
        TerminalClassification::Success => Value::Null,
        TerminalClassification::Failure { kind, message } => {
            json!({"kind": kind, "message": message})
        }
    };
    let terminal = TerminalRecord {
        call_id: context.litellm_call_id,
        trace_id: context.trace_id,
        attempt: context.attempt,
        call_type: context.call_type,
        model,
        provider: context.custom_llm_provider,
        timing: CallbackTiming::new(start_time, services.now()),
        usage: observation.usage,
        cost_inputs: CostInputs {
            response_cost: context.response_cost,
            metadata: context.metadata,
        },
        classification,
        projection: RouteProjection::ResponsesWs { value: projection },
    };
    let _ = services.dispatch(&terminal).await;
    Ok(match result {
        Ok(()) => ExecutedCall::Success {
            response: (),
            terminal,
        },
        Err(error) => ExecutedCall::Failure { error, terminal },
    })
}

async fn splice<In, Out>(
    upstream: Upstream,
    model: &str,
    first_frame: Option<ResponsesWsEvent>,
    idle_timeout: Duration,
    instrumentation: &ResponsesWsInstrumentation,
    mut client_in: In,
    mut client_out: Out,
) -> Result<(), Error>
where
    In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let (mut upstream_tx, mut upstream_rx) = upstream.split();
    if let Some(event) = first_frame {
        send_provider_event(&mut upstream_tx, &event, model).await?;
    }
    loop {
        tokio::select! {
            event = client_in.next() => {
                let Some(event) = event else { return Ok(()) };
                send_provider_event(&mut upstream_tx, &event, model).await?;
            }
            message = upstream_rx.next() => {
                let Some(message) = message else { return Ok(()) };
                match message.map_err(ws_transport_error)? {
                    Message::Text(text) => {
                        let event = serde_json::from_str::<ResponsesWsEvent>(&text)
                            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
                        instrumentation.observe(&event);
                        for outbound in OPENAI_RESPONSES_WS_CONFIG.transform_ws_response(&event, model)?.events {
                            client_out.send(outbound).await
                                .map_err(|error| Error::Network(error.to_string()))?;
                        }
                    }
                    Message::Close(_) => return Ok(()),
                    _ => {}
                }
            }
            _ = tokio::time::sleep(idle_timeout) => return Ok(()),
        }
    }
}

async fn send_provider_event(
    upstream: &mut futures_util::stream::SplitSink<Upstream, Message>,
    event: &ResponsesWsEvent,
    model: &str,
) -> Result<(), Error> {
    for outbound in OPENAI_RESPONSES_WS_CONFIG
        .transform_ws_request(event, model)?
        .events
    {
        let payload = serde_json::to_string(&outbound)
            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
        upstream
            .send(Message::Text(payload.into()))
            .await
            .map_err(ws_transport_error)?;
    }
    Ok(())
}

async fn dial_upstream(
    model: &str,
    api_key: &str,
    api_base: Option<&str>,
) -> Result<Upstream, Error> {
    let url = OPENAI_RESPONSES_WS_CONFIG.complete_websocket_url(api_base, model);
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
    result.map(|(socket, _)| socket).map_err(ws_handshake_error)
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

fn resolve_api_key(api_key: Option<&str>) -> Result<String, Error> {
    api_key
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(OPENAI_API_KEY_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .ok_or_else(|| Error::Auth(MISSING_KEY_MESSAGE.to_string()))
}

pub fn complete_websocket_url(api_base: Option<&str>, model: &str, model_in_url: bool) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let (base, query) = base
        .split_once('?')
        .map_or((base, None), |(base, query)| (base, Some(query)));
    let response_url = format!("{}{}", base.trim_end_matches('/'), OPENAI_RESPONSES_PATH);
    let response_url = response_url
        .strip_prefix("https://")
        .map(|rest| format!("wss://{rest}"))
        .or_else(|| {
            response_url
                .strip_prefix("http://")
                .map(|rest| format!("ws://{rest}"))
        })
        .unwrap_or(response_url);
    let url = query.map_or_else(
        || response_url.clone(),
        |query| format!("{response_url}?{query}"),
    );
    if !model_in_url
        || query.is_some_and(|query| {
            query
                .split('&')
                .any(|part| part.split('=').next() == Some("model"))
        })
    {
        return url;
    }
    format!(
        "{url}{}model={}",
        if query.is_some() { "&" } else { "?" },
        percent_encode(model)
    )
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                format!("{}", byte as char)
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

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
        ResponsesWsEventType::ResponseCreated
            | ResponsesWsEventType::ResponseCompleted
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

    async fn mock_provider() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            if let Some(Ok(Message::Text(text))) = socket.next().await {
                let request: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(request["model"], "authorized");
                socket.send(Message::Text(json!({"type":"response.completed","response":{"id":"resp-1","model":"authorized","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}).to_string().into())).await.unwrap();
                socket.close(None).await.unwrap();
            }
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn core_owns_splice_transformation_and_one_terminal() {
        let (api_base, server) = mock_provider().await;
        let services = Services::default();
        let (client_tx, client_rx) = futures_channel::mpsc::unbounded();
        let (output_tx, mut output_rx) = futures_channel::mpsc::unbounded();
        client_tx
            .unbounded_send(event(json!({"type":"response.create","model":"wrong"})))
            .unwrap();
        let result = responses_websocket(
            &services,
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
        assert_eq!(terminals[0].usage.total_tokens, 3);
        server.await.unwrap();
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
        let services = Services::default();
        let (_, input): (
            _,
            futures_channel::mpsc::UnboundedReceiver<ResponsesWsEvent>,
        ) = futures_channel::mpsc::unbounded();
        let (output, _) = futures_channel::mpsc::unbounded();
        let error = responses_websocket(
            &services,
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
        let services = Services::default();
        let (client_tx, input) = futures_channel::mpsc::unbounded();
        let (output, _) = futures_channel::mpsc::unbounded();
        let result = responses_websocket(
            &services,
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
