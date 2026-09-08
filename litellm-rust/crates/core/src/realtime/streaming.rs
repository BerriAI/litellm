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
    CallLifecycleContext, Clock, CostInputs, ExecutedCall, RouteProjection, TerminalClassification,
    TerminalDispatcher, TerminalRecord,
};
use crate::providers::dispatch::realtime_provider_config;
use crate::realtime::transformation::RealtimeProviderConfig;
use crate::realtime::types::RealtimeEvent;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

type Upstream = WebSocketStream<MaybeTlsStream<TcpStream>>;
static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

#[derive(Clone)]
pub struct RealtimeConnectionSpec {
    config: &'static (dyn RealtimeProviderConfig + Sync),
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
        let model = model.into();
        let (model, config) = realtime_provider_config(&model)?;
        Ok(Self {
            config,
            model: model.to_string(),
            api_key: config.resolve_api_key(api_key, &|key| std::env::var(key).ok())?,
            api_base: api_base.map(str::to_string),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Eq for RealtimeConnectionSpec {}

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
    connection: RealtimeConnectionSpec,
    upstream: Upstream,
    session_created: RealtimeEvent,
}

impl WarmConnection {
    pub fn is_live(&mut self) -> bool {
        let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
        matches!(
            Pin::new(&mut self.upstream).poll_next(&mut context),
            Poll::Pending
        )
    }
}

pub struct RealtimeRequest {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
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
        connection: connection.clone(),
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
    let model = request.model;
    let connection = RealtimeConnectionSpec::new(
        model.clone(),
        request.api_key.as_deref(),
        request.api_base.as_deref(),
    );
    let connection = match (connection, request.warm) {
        (Ok(connection), Some(warm)) if warm.connection == connection => Ok(warm),
        (Ok(_), Some(_)) => Err(Error::InvalidRequest(
            "realtime warm connection does not match the requested provider connection".to_string(),
        )),
        (Ok(connection), None) => dial_upstream(&connection)
            .await
            .map(|upstream| WarmConnection {
                connection,
                upstream,
                session_created: empty_event(),
            }),
        (Err(error), _) => Err(error),
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
        Err(error) => Err(error.into()),
    };
    let classification = match &result {
        Ok(()) => TerminalClassification::Success,
        Err(failure) => TerminalClassification::Failure {
            kind: failure.kind.to_string(),
            message: failure.error.to_string(),
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
        Err(failure) => ExecutedCall::Failure {
            error: failure.error,
            terminal,
        },
    }
}

async fn splice<In, Out>(
    connection: WarmConnection,
    model: &str,
    idle_timeout: Duration,
    observation: &mut RealtimeObservation,
    mut client_in: In,
    mut client_out: Out,
) -> Result<(), RealtimeFailure>
where
    In: Stream<Item = RealtimeEvent> + Unpin + Send,
    Out: Sink<RealtimeEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let WarmConnection {
        connection,
        upstream,
        session_created,
    } = connection;
    let config = connection.config;
    let (mut upstream_tx, mut upstream_rx) = upstream.split();
    if !session_created.event_type.is_empty() {
        observation.observe(&session_created);
        send_client_event(config, &mut client_out, &session_created, model).await?;
    }
    loop {
        tokio::select! {
            event = client_in.next() => {
                let Some(event) = event else {
                    return observation.settle(
                        "Cancelled",
                        "realtime client disconnected before provider completion",
                    );
                };
                for outbound in config.transform_realtime_request(&event, model)?.events {
                    let payload = serde_json::to_string(&outbound)
                        .map_err(|error| Error::InvalidResponse(error.to_string()))?;
                    upstream_tx.send(Message::Text(payload)).await.map_err(ws_transport_error)?;
                }
                observation.observe_client(&event);
            }
            message = upstream_rx.next() => {
                let Some(message) = message else {
                    return observation.settle(
                        "NetworkError",
                        "realtime provider closed before completion",
                    );
                };
                match message.map_err(ws_transport_error)? {
                    Message::Text(text) => {
                        let event = serde_json::from_str::<RealtimeEvent>(&text)
                            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
                        observation.observe(&event);
                        send_client_event(config, &mut client_out, &event, model).await?;
                        if event.event_type == "error" || response_failed(&event) {
                            return Err(RealtimeFailure::new(
                                "ProviderError",
                                Error::InvalidResponse(provider_error_message(&event)),
                            ));
                        }
                    }
                    Message::Close(_) => {
                        return observation.settle(
                            "NetworkError",
                            "realtime provider closed before completion",
                        );
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep(idle_timeout) => {
                return Err(RealtimeFailure::new(
                    "Timeout",
                    Error::Network("realtime session idle timeout".to_string()),
                ));
            }
        }
    }
}

struct RealtimeFailure {
    kind: &'static str,
    error: Error,
}

impl RealtimeFailure {
    fn new(kind: &'static str, error: Error) -> Self {
        Self { kind, error }
    }
}

impl From<Error> for RealtimeFailure {
    fn from(error: Error) -> Self {
        Self {
            kind: error_kind(&error),
            error,
        }
    }
}

fn response_failed(event: &RealtimeEvent) -> bool {
    event.event_type == "response.done"
        && event
            .data
            .get("response")
            .and_then(|response| response.get("status"))
            .and_then(Value::as_str)
            .is_some_and(|status| status != "completed")
}

fn provider_error_message(event: &RealtimeEvent) -> String {
    event
        .data
        .get("error")
        .and_then(|error| error.get("message"))
        .or_else(|| {
            event
                .data
                .get("response")
                .and_then(|response| response.get("status_details"))
                .and_then(|details| details.get("error"))
                .and_then(|error| error.get("message"))
        })
        .and_then(Value::as_str)
        .unwrap_or("realtime provider reported an error")
        .to_string()
}

async fn send_client_event<Out>(
    config: &(dyn RealtimeProviderConfig + Sync),
    client_out: &mut Out,
    event: &RealtimeEvent,
    model: &str,
) -> Result<(), Error>
where
    Out: Sink<RealtimeEvent> + Unpin,
    Out::Error: std::fmt::Display,
{
    for outbound in config.transform_realtime_response(event, model)?.events {
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
    let url = connection
        .config
        .complete_url(connection.api_base.as_deref(), connection.model.as_str());
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
    completed_response: bool,
    pending_responses: usize,
}

impl RealtimeObservation {
    fn new(call_id: String, model: String) -> Self {
        Self {
            call_id,
            model,
            usage: Usage::default(),
            completed_response: false,
            pending_responses: 0,
        }
    }

    fn settle(&self, kind: &'static str, message: &str) -> Result<(), RealtimeFailure> {
        if self.completed_response && self.pending_responses == 0 {
            Ok(())
        } else if kind == "Cancelled" {
            Err(RealtimeFailure::new(
                kind,
                Error::InvalidRequest(message.to_string()),
            ))
        } else {
            Err(RealtimeFailure::new(
                kind,
                Error::Network(message.to_string()),
            ))
        }
    }

    fn observe_client(&mut self, event: &RealtimeEvent) {
        if event.event_type == "response.create" {
            self.pending_responses += 1;
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
        if !response_failed(event) {
            self.completed_response = true;
            self.pending_responses = self.pending_responses.saturating_sub(1);
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
        let input = usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
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

    async fn scripted_provider(events: Vec<Value>, close_after_events: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let events = events.clone();
                tokio::spawn(async move {
                    let mut socket = accept_async(stream).await.unwrap();
                    socket.send(Message::Text(json!({"type":"session.created","session":{"id":"sess-core","model":"upstream-model"}}).to_string())).await.unwrap();
                    while let Some(Ok(Message::Text(text))) = socket.next().await {
                        let event: RealtimeEvent = serde_json::from_str(&text).unwrap();
                        if event.event_type == "response.create" {
                            for event in &events {
                                if socket.send(Message::Text(event.to_string())).await.is_err() {
                                    return;
                                }
                            }
                            if close_after_events {
                                let _ = socket.close(None).await;
                            }
                        }
                    }
                });
            }
        });
        format!("ws://{address}")
    }

    async fn provider() -> String {
        scripted_provider(
            vec![
                json!({"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}),
                json!({"type":"response.done","response":{"usage":{"input_tokens":7,"output_tokens":11}}}),
            ],
            true,
        )
        .await
    }

    async fn execute_scenario(
        events: Vec<Value>,
        close_after_events: bool,
        send_response_create: bool,
        disconnect_client: bool,
        idle_timeout: Duration,
    ) -> (ExecutedCall<(), Error>, Vec<TerminalRecord>) {
        let base = scripted_provider(events, close_after_events).await;
        let services = Services::default();
        let (input_tx, input) = mpsc::unbounded();
        if send_response_create {
            input_tx
                .unbounded_send(serde_json::from_value(json!({"type":"response.create"})).unwrap())
                .unwrap();
        }
        if disconnect_client {
            drop(input_tx);
        }
        let (output, _output_rx) = mpsc::unbounded();
        let result = realtime(
            &services,
            RealtimeRequest {
                model: "requested".to_string(),
                api_key: Some("key".to_string()),
                api_base: Some(base),
                warm: None,
                idle_timeout: Some(idle_timeout),
            },
            CallLifecycleContext::new("realtime", "requested", "openai", "fallback"),
            input,
            output,
        )
        .await;
        let terminals = services.terminals.into_inner().unwrap();
        (result, terminals)
    }

    async fn execute(warm: bool) -> (ExecutedCall<(), Error>, Vec<RealtimeEvent>, usize) {
        let base = provider().await;
        let spec = RealtimeConnectionSpec::new("requested", Some("key"), Some(&base)).unwrap();
        let services = Services::default();
        let warm = if warm {
            Some(warmup(&spec).await.unwrap())
        } else {
            None
        };
        assert!(services.terminals.lock().unwrap().is_empty());
        let (input_tx, input) = mpsc::unbounded();
        let (output, mut output_rx) = mpsc::unbounded();
        input_tx.unbounded_send(serde_json::from_value(json!({"type":"response.done","response":{"usage":{"input_tokens":1000,"output_tokens":1000,"total_tokens":2000}}})).unwrap()).unwrap();
        input_tx
            .unbounded_send(serde_json::from_value(json!({"type":"response.create"})).unwrap())
            .unwrap();
        let result = realtime(
            &services,
            RealtimeRequest {
                model: spec.model.clone(),
                api_key: Some("key".to_string()),
                api_base: Some(base),
                warm,
                idle_timeout: Some(Duration::from_secs(1)),
            },
            CallLifecycleContext::new("realtime", "requested", "openai", "fallback"),
            input,
            output,
        )
        .await;
        let mut events = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_millis(10), output_rx.next()).await
        {
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
            let ExecutedCall::Success { terminal, .. } = result else {
                panic!("session failed")
            };
            assert_eq!(terminal.classification, TerminalClassification::Success);
            assert_eq!(terminal.call_id, "sess-core");
            assert_eq!(terminal.model, "upstream-model");
            assert_eq!(
                terminal.usage,
                Usage {
                    prompt_tokens: 9,
                    completion_tokens: 14,
                    total_tokens: 23
                }
            );
        }
    }

    #[tokio::test]
    async fn client_disconnect_before_provider_completion_is_cancelled_once() {
        let (result, terminals) =
            execute_scenario(Vec::new(), false, false, true, Duration::from_secs(1)).await;

        assert!(matches!(result, ExecutedCall::Failure { .. }));
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            &terminals[0].classification,
            TerminalClassification::Failure { kind, .. } if kind == "Cancelled"
        ));
    }

    #[tokio::test]
    async fn idle_timeout_before_provider_completion_fails_once() {
        let (result, terminals) =
            execute_scenario(Vec::new(), false, false, false, Duration::from_millis(20)).await;

        assert!(matches!(result, ExecutedCall::Failure { .. }));
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            &terminals[0].classification,
            TerminalClassification::Failure { kind, .. } if kind == "Timeout"
        ));
    }

    #[tokio::test]
    async fn provider_error_event_fails_once() {
        let (result, terminals) = execute_scenario(
            vec![json!({"type":"error","error":{"message":"provider rejected event"}})],
            false,
            true,
            false,
            Duration::from_secs(1),
        )
        .await;

        assert!(matches!(result, ExecutedCall::Failure { .. }));
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            &terminals[0].classification,
            TerminalClassification::Failure { kind, message }
                if kind == "ProviderError" && message.contains("provider rejected event")
        ));
    }

    #[tokio::test]
    async fn provider_close_before_response_done_fails_once() {
        let (result, terminals) =
            execute_scenario(Vec::new(), true, true, false, Duration::from_secs(1)).await;

        assert!(matches!(result, ExecutedCall::Failure { .. }));
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            &terminals[0].classification,
            TerminalClassification::Failure { kind, .. } if kind == "NetworkError"
        ));
    }

    #[tokio::test]
    async fn warmup_success_and_failure_dispatch_nothing() {
        let services = Services::default();
        let base = provider().await;
        let good = RealtimeConnectionSpec::new("model", Some("key"), Some(&base)).unwrap();
        assert!(warmup(&good).await.is_ok());
        let bad =
            RealtimeConnectionSpec::new("model", Some("key"), Some("ws://127.0.0.1:1")).unwrap();
        assert!(warmup(&bad).await.is_err());
        assert!(services.terminals.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn fresh_dial_failure_returns_and_dispatches_one_terminal() {
        let services = Services::default();
        let (_, input) = mpsc::unbounded();
        let (output, _) = mpsc::unbounded();
        let result = realtime(
            &services,
            RealtimeRequest {
                model: "model".to_string(),
                api_key: Some("key".to_string()),
                api_base: Some("ws://127.0.0.1:1".to_string()),
                warm: None,
                idle_timeout: None,
            },
            CallLifecycleContext::new("realtime", "model", "openai", "call-failure"),
            input,
            output,
        )
        .await;

        assert!(matches!(result, ExecutedCall::Failure { .. }));
        let terminals = services.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            terminals[0].classification,
            TerminalClassification::Failure { .. }
        ));
    }
}
