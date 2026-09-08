use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
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
use crate::integrations::types::Usage;
use crate::lifecycle::{
    CallLifecycleContext, Clock, CostInputs, ExecutedCall, RouteProjection,
    TerminalClassification, TerminalDispatcher, TerminalRecord,
};
use crate::providers::openai::realtime::transformation::OPENAI_REALTIME_CONFIG;
use crate::realtime::transformation::RealtimeProviderConfig;
use crate::realtime::types::RealtimeEvent;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const MISSING_KEY_MESSAGE: &str = "Missing OpenAI API Key - a realtime call is being made but no key was passed via params or the OPENAI_API_KEY environment variable";

type Upstream = WebSocketStream<MaybeTlsStream<TcpStream>>;
static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

#[derive(Clone, Eq)]
pub struct RealtimeConnectionSpec {
    model: String,
    api_key: String,
    api_base: Option<String>,
}

impl RealtimeConnectionSpec {
    pub fn new(
        model: impl Into<String>,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> Result<Self, Error> {
        Ok(Self {
            model: model.into(),
            api_key: resolve_api_key(api_key)?,
            api_base: api_base.map(str::to_string),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

impl PartialEq for RealtimeConnectionSpec {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.api_key == other.api_key
            && self.api_base == other.api_base
    }
}

impl Hash for RealtimeConnectionSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.api_key.hash(state);
        self.api_base.hash(state);
    }
}

impl std::fmt::Debug for RealtimeConnectionSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealtimeConnectionSpec")
            .field("model", &self.model)
            .field("api_key", &"[REDACTED]")
            .field("api_base", &self.api_base)
            .finish()
    }
}

pub struct WarmConnection {
    upstream: Upstream,
    session_created: RealtimeEvent,
}

impl WarmConnection {
    pub fn is_live(&mut self) -> bool {
        let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
        matches!(Pin::new(&mut self.upstream).poll_next(&mut context), Poll::Pending)
    }
}

pub struct RealtimeRequest {
    pub connection: RealtimeConnectionSpec,
    pub warm: Option<WarmConnection>,
    pub idle_timeout: Option<Duration>,
}

pub async fn warmup(connection: &RealtimeConnectionSpec) -> Result<WarmConnection, Error> {
    let mut upstream = dial_upstream(connection).await?;
    let session_created = read_event(&mut upstream).await?;
    if session_created.event_type != "session.created" {
        return Err(Error::InvalidResponse(format!(
            "expected session.created during realtime warmup, received {}",
            session_created.event_type
        )));
    }
    Ok(WarmConnection {
        upstream,
        session_created,
    })
}

pub async fn realtime<S, In, Out>(
    services: &S,
    request: RealtimeRequest,
    context: CallLifecycleContext,
    client_in: In,
    client_out: Out,
) -> ExecutedCall<(), Error>
where
    S: TerminalDispatcher + Clock,
    In: Stream<Item = RealtimeEvent> + Unpin + Send,
    Out: Sink<RealtimeEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let start_time = services.now();
    let model = request.connection.model.clone();
    let connection = match request.warm {
        Some(warm) => Ok(warm),
        None => dial_upstream(&request.connection)
            .await
            .map(|upstream| WarmConnection {
                upstream,
                session_created: empty_event(),
            }),
    };
    let mut observation = RealtimeObservation::new(context.litellm_call_id.clone(), model.clone());
    let result = match connection {
        Ok(connection) => {
            splice(
                connection,
                &model,
                request.idle_timeout.unwrap_or(IDLE_TIMEOUT),
                &mut observation,
                client_in,
                client_out,
            )
            .await
        }
        Err(error) => Err(error),
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
        call_id: observation.call_id,
        trace_id: context.trace_id,
        attempt: context.attempt,
        call_type: context.call_type,
        model: observation.model,
        provider: context.custom_llm_provider,
        timing: CallbackTiming::new(start_time, services.now()),
        usage: observation.usage,
        cost_inputs: CostInputs {
            response_cost: context.response_cost,
            metadata: context.metadata,
        },
        classification,
        projection: RouteProjection::Realtime { value: projection },
    };
    let _ = services.dispatch(&terminal).await;
    match result {
        Ok(()) => ExecutedCall::Success {
            response: (),
            terminal,
        },
        Err(error) => ExecutedCall::Failure { error, terminal },
    }
}

async fn splice<In, Out>(
    connection: WarmConnection,
    model: &str,
    idle_timeout: Duration,
    observation: &mut RealtimeObservation,
    mut client_in: In,
    mut client_out: Out,
) -> Result<(), Error>
where
    In: Stream<Item = RealtimeEvent> + Unpin + Send,
    Out: Sink<RealtimeEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let WarmConnection {
        upstream,
        session_created,
    } = connection;
    let (mut upstream_tx, mut upstream_rx) = upstream.split();
    if !session_created.event_type.is_empty() {
        observation.observe(&session_created);
        send_client_event(&mut client_out, &session_created, model).await?;
    }
    loop {
        tokio::select! {
            event = client_in.next() => {
                let Some(event) = event else { return Ok(()) };
                for outbound in OPENAI_REALTIME_CONFIG.transform_realtime_request(&event, model)?.events {
                    let payload = serde_json::to_string(&outbound)
                        .map_err(|error| Error::InvalidResponse(error.to_string()))?;
                    upstream_tx.send(Message::Text(payload.into())).await.map_err(ws_transport_error)?;
                }
            }
            message = upstream_rx.next() => {
                let Some(message) = message else { return Ok(()) };
                match message.map_err(ws_transport_error)? {
                    Message::Text(text) => {
                        let event = serde_json::from_str::<RealtimeEvent>(&text)
                            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
                        observation.observe(&event);
                        send_client_event(&mut client_out, &event, model).await?;
                    }
                    Message::Close(_) => return Ok(()),
                    _ => {}
                }
            }
            _ = tokio::time::sleep(idle_timeout) => return Ok(()),
        }
    }
}

async fn send_client_event<Out>(
    client_out: &mut Out,
    event: &RealtimeEvent,
    model: &str,
) -> Result<(), Error>
where
    Out: Sink<RealtimeEvent> + Unpin,
    Out::Error: std::fmt::Display,
{
    for outbound in OPENAI_REALTIME_CONFIG
        .transform_realtime_response(event, model)?
        .events
    {
        client_out
            .send(outbound)
            .await
            .map_err(|error| Error::Network(error.to_string()))?;
    }
    Ok(())
}

async fn read_event(upstream: &mut Upstream) -> Result<RealtimeEvent, Error> {
    loop {
        let message = upstream
            .next()
            .await
            .ok_or_else(|| Error::Network("upstream closed before first event".to_string()))?
            .map_err(ws_transport_error)?;
        match message {
            Message::Text(text) => {
                return serde_json::from_str(&text)
                    .map_err(|error| Error::InvalidResponse(error.to_string()));
            }
            Message::Close(_) => {
                return Err(Error::Network(
                    "upstream closed before first event".to_string(),
                ));
            }
            _ => {}
        }
    }
}

async fn dial_upstream(connection: &RealtimeConnectionSpec) -> Result<Upstream, Error> {
    let url = OPENAI_REALTIME_CONFIG.complete_url(
        connection.api_base.as_deref(),
        connection.model.as_str(),
    );
    let mut request = url.into_client_request().map_err(ws_transport_error)?;
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", connection.api_key))
            .map_err(|error| Error::Auth(error.to_string()))?,
    );
    let connector = match request.uri().scheme_str() {
        Some("wss") => Some(Connector::Rustls(tls_config()?)),
        _ => None,
    };
    let result = tokio::time::timeout(
        CONNECT_TIMEOUT,
        connect_async_tls_with_config(request, None, false, connector),
    )
    .await
    .map_err(|_| Error::Connect("realtime WebSocket connection timed out".to_string()))?;
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
    let config = ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|error| Error::Connect(error.to_string()))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    let config = Arc::new(config);
    Ok(Arc::clone(TLS_CONFIG.get_or_init(|| config)))
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

fn empty_event() -> RealtimeEvent {
    RealtimeEvent {
        event_type: String::new(),
        data: Default::default(),
    }
}

struct RealtimeObservation {
    call_id: String,
    model: String,
    usage: Usage,
}

impl RealtimeObservation {
    fn new(call_id: String, model: String) -> Self {
        Self {
            call_id,
            model,
            usage: Usage::default(),
        }
    }

    fn observe(&mut self, event: &RealtimeEvent) {
        if event.event_type == "session.created" {
            let session = event.data.get("session").and_then(Value::as_object);
            if let Some(id) = session
                .and_then(|value| value.get("id"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            {
                self.call_id = id.to_string();
            }
            if let Some(model) = session
                .and_then(|value| value.get("model"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            {
                self.model = model.to_string();
            }
            return;
        }
        if event.event_type != "response.done" {
            return;
        }
        let Some(usage) = event
            .data
            .get("response")
            .and_then(Value::as_object)
            .and_then(|value| value.get("usage"))
            .and_then(Value::as_object)
        else {
            return;
        };
        let input = usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.usage.prompt_tokens += input;
        self.usage.completion_tokens += output;
        self.usage.total_tokens += usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(input + output);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_channel::mpsc;
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    use super::*;
    use crate::integrations::custom_logger::{LogError, LogFuture};
    use crate::lifecycle::{CallLifecycleContext, Clock};

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

    async fn provider() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut socket = accept_async(stream).await.unwrap();
                    socket.send(Message::Text(json!({"type":"session.created","session":{"id":"sess-core","model":"upstream-model"}}).to_string().into())).await.unwrap();
                    while let Some(Ok(Message::Text(text))) = socket.next().await {
                        let event: RealtimeEvent = serde_json::from_str(&text).unwrap();
                        if event.event_type == "response.create" {
                            socket.send(Message::Text(json!({"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}).to_string().into())).await.unwrap();
                            socket.send(Message::Text(json!({"type":"response.done","response":{"usage":{"input_tokens":7,"output_tokens":11}}}).to_string().into())).await.unwrap();
                            socket.close(None).await.unwrap();
                        }
                    }
                });
            }
        });
        format!("ws://{address}")
    }

    async fn execute(warm: bool) -> (ExecutedCall<(), Error>, Vec<RealtimeEvent>, usize) {
        let base = provider().await;
        let spec = RealtimeConnectionSpec::new("requested", Some("key"), Some(&base)).unwrap();
        let services = Services::default();
        let warm = if warm { Some(warmup(&spec).await.unwrap()) } else { None };
        assert!(services.terminals.lock().unwrap().is_empty());
        let (input_tx, input) = mpsc::unbounded();
        let (output, mut output_rx) = mpsc::unbounded();
        input_tx.unbounded_send(serde_json::from_value(json!({"type":"response.done","response":{"usage":{"input_tokens":1000,"output_tokens":1000,"total_tokens":2000}}})).unwrap()).unwrap();
        input_tx.unbounded_send(serde_json::from_value(json!({"type":"response.create"})).unwrap()).unwrap();
        let result = realtime(
            &services,
            RealtimeRequest { connection: spec, warm, idle_timeout: Some(Duration::from_secs(1)) },
            CallLifecycleContext::new("realtime", "requested", "openai", "fallback"),
            input,
            output,
        ).await;
        let mut events = Vec::new();
        while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(10), output_rx.next()).await {
            events.push(event);
        }
        let count = services.terminals.lock().unwrap().len();
        (result, events, count)
    }

    #[tokio::test]
    async fn fresh_and_warm_sessions_share_identity_usage_and_exactly_once_terminal() {
        for warm in [false, true] {
            let (result, events, count) = execute(warm).await;
            assert_eq!(count, 1);
            assert_eq!(events.first().unwrap().event_type, "session.created");
            let ExecutedCall::Success { terminal, .. } = result else { panic!("session failed") };
            assert_eq!(terminal.call_id, "sess-core");
            assert_eq!(terminal.model, "upstream-model");
            assert_eq!(terminal.usage, Usage { prompt_tokens: 9, completion_tokens: 14, total_tokens: 23 });
        }
    }

    #[tokio::test]
    async fn warmup_success_and_failure_dispatch_nothing() {
        let services = Services::default();
        let base = provider().await;
        let good = RealtimeConnectionSpec::new("model", Some("key"), Some(&base)).unwrap();
        assert!(warmup(&good).await.is_ok());
        let bad = RealtimeConnectionSpec::new("model", Some("key"), Some("ws://127.0.0.1:1")).unwrap();
        assert!(warmup(&bad).await.is_err());
        assert!(services.terminals.lock().unwrap().is_empty());
    }
}
