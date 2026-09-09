use std::future::{Ready, ready};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures_util::{StreamExt, stream};
use litellm_core::Error;
use litellm_core::integrations::custom_logger::{LogError, LogFuture};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, Clock, ModerationHooks, PreCallHooks,
    TerminalClassification, TerminalDispatcher, TerminalRecord,
};
use litellm_core::messages::lifecycle::Options;
use litellm_core::messages::types::MessagesRequest;
use litellm_core::providers::auth::Environment;
use litellm_core::runtime::{
    CallServices, HttpFuture, HttpRequest, HttpStreamFuture, HttpStreamResponse, HttpTransport,
    LiteLlm, MessagesRuntimeServices,
};
use serde_json::json;

struct Tracking {
    opened: AtomicUsize,
    session_drops: AtomicUsize,
    service_drops: AtomicUsize,
    terminals: Mutex<Vec<TerminalRecord>>,
}

impl Tracking {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            opened: AtomicUsize::new(0),
            session_drops: AtomicUsize::new(0),
            service_drops: AtomicUsize::new(0),
            terminals: Mutex::new(Vec::new()),
        })
    }
}

struct Session {
    tracking: Arc<Tracking>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.tracking.session_drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl Clock for Session {
    fn now(&self) -> f64 {
        10.0
    }
}

impl PreCallHooks<MessagesRequest> for Session {
    type PreCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl ModerationHooks<MessagesRequest> for Session {
    type ModerationFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::ModerationFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl TerminalDispatcher for Session {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async move {
            self.tracking
                .terminals
                .lock()
                .unwrap()
                .push(terminal.clone());
            Ok::<(), LogError>(())
        })
    }
}

#[derive(Clone)]
struct Calls {
    tracking: Arc<Tracking>,
}

impl CallServices for Calls {
    type Bindings = u64;
    type Session = Session;
    type OpenFuture<'a> = Ready<Result<Session, Error>>;

    fn open<'a>(&'a self, _: CallLifecycleContext, _: Self::Bindings) -> Self::OpenFuture<'a> {
        self.tracking.opened.fetch_add(1, Ordering::Relaxed);
        ready(Ok(Session {
            tracking: self.tracking.clone(),
        }))
    }
}

struct Transport {
    response: Mutex<Option<Result<HttpStreamResponse, Error>>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl HttpTransport for Transport {
    fn execute(&self, _: HttpRequest) -> HttpFuture<'_> {
        Box::pin(ready(Err(Error::Unsupported("buffered test transport"))))
    }

    fn execute_stream(&self, request: HttpRequest) -> HttpStreamFuture<'_> {
        self.requests.lock().unwrap().push(request);
        let response = self.response.lock().unwrap().take().unwrap();
        Box::pin(ready(response))
    }
}

struct Services {
    transport: Transport,
    calls: Calls,
    tracking: Arc<Tracking>,
}

impl Drop for Services {
    fn drop(&mut self) {
        self.tracking.service_drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl Environment for Services {
    fn environment(&self, _: &str) -> Option<String> {
        None
    }
}

impl MessagesRuntimeServices for Services {
    type Transport = Transport;
    type Calls = Calls;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

fn request() -> MessagesRequest {
    MessagesRequest {
        model: "anthropic/claude-test".to_string(),
        body: json!({
            "model": "claude-test",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }),
        api_key: Some("test-key".to_string()),
        api_base: Some("https://recording.invalid/v1/messages".to_string()),
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    }
}

fn context() -> CallLifecycleContext {
    CallLifecycleContext::new("messages", "claude-test", "anthropic", "call-1")
}

fn client(
    response: Result<HttpStreamResponse, Error>,
    tracking: Arc<Tracking>,
) -> LiteLlm<Services> {
    LiteLlm::from_services(Services {
        transport: Transport {
            response: Mutex::new(Some(response)),
            requests: Mutex::new(Vec::new()),
        },
        calls: Calls {
            tracking: tracking.clone(),
        },
        tracking,
    })
}

fn response(stream: litellm_core::lifecycle::BytesStream) -> HttpStreamResponse {
    HttpStreamResponse {
        status: 200,
        content_type: Some("text/event-stream".to_string()),
        cache_control: Some("no-cache".to_string()),
        stream,
    }
}

#[tokio::test]
async fn completion_retains_application_and_session_until_terminal_delivery() {
    let tracking = Tracking::new();
    let events = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let client = client(
        Ok(response(Box::pin(stream::iter([Ok(Bytes::from_static(
            events,
        ))])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(request(), Options::default(), context(), 1)
        .await
        .unwrap();

    assert_eq!(tracking.opened.load(Ordering::Relaxed), 1);
    drop(client);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 0);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 0);

    let completion = call.completion.register();
    let bytes = call
        .stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .concat();
    assert_eq!(bytes, events);
    let terminal = completion.await.unwrap();

    assert_eq!(terminal.classification, TerminalClassification::Success);
    assert_eq!(terminal.usage.prompt_tokens, 5);
    assert_eq!(terminal.usage.completion_tokens, 4);
    assert_eq!(terminal.usage.total_tokens, 9);
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn unregistered_completion_still_delivers_terminal_before_cleanup() {
    let tracking = Tracking::new();
    let client = client(
        Ok(response(Box::pin(stream::iter([Ok(Bytes::from_static(
            b"data: {\"type\":\"message_stop\"}\n\n",
        ))])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(request(), Options::default(), context(), 5)
        .await
        .unwrap();
    drop(client);
    let stream = call.stream;
    drop(call.completion);
    stream.collect::<Vec<_>>().await;

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while tracking.terminals.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn partial_consumption_and_early_drop_cancel_once_and_release_owners() {
    let tracking = Tracking::new();
    let client = client(
        Ok(response(Box::pin(stream::iter([
            Ok(Bytes::from_static(b"data: first\n\n")),
            Ok(Bytes::from_static(b"data: second\n\n")),
        ])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(request(), Options::default(), context(), 2)
        .await
        .unwrap();
    drop(client);
    let completion = call.completion.register();
    let mut stream = call.stream;

    assert_eq!(stream.next().await.unwrap().unwrap(), "data: first\n\n");
    drop(stream);
    let terminal = completion.await.unwrap();

    assert!(matches!(
        terminal.classification,
        TerminalClassification::Failure { ref kind, .. } if kind == "Cancelled"
    ));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn provider_failure_dispatches_before_releasing_the_session() {
    let tracking = Tracking::new();
    let client = client(
        Ok(HttpStreamResponse {
            status: 500,
            content_type: Some("application/json".to_string()),
            cache_control: None,
            stream: Box::pin(stream::iter([Ok(Bytes::from_static(b"failed"))])),
        }),
        tracking.clone(),
    );
    let result = client
        .messages_stream_with(request(), Options::default(), context(), 3)
        .await;
    let error = match result {
        Ok(_) => panic!("provider rejects the stream"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::Http { status: 500, .. }));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    drop(client);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn cancelled_consumer_releases_stream_owners() {
    let tracking = Tracking::new();
    let client = client(Ok(response(Box::pin(stream::pending()))), tracking.clone());
    let call = client
        .messages_stream_with(request(), Options::default(), context(), 4)
        .await
        .unwrap();
    drop(client);
    let completion = call.completion.register();
    let consumer = tokio::spawn(async move {
        let mut stream = call.stream;
        stream.next().await
    });
    tokio::task::yield_now().await;
    consumer.abort();
    assert!(consumer.await.unwrap_err().is_cancelled());
    let terminal = completion.await.unwrap();

    assert!(matches!(
        terminal.classification,
        TerminalClassification::Failure { ref kind, .. } if kind == "Cancelled"
    ));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn stream_failure_dispatches_once_and_releases_owners() {
    let tracking = Tracking::new();
    let client = client(
        Ok(response(Box::pin(stream::iter([Err(Error::Network(
            "stream failed".to_string(),
        ))])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(request(), Options::default(), context(), 6)
        .await
        .unwrap();
    drop(client);
    let completion = call.completion.register();
    let chunks = call.stream.collect::<Vec<_>>().await;
    let terminal = completion.await.unwrap();

    assert!(chunks.iter().any(Result::is_err));
    assert!(matches!(
        terminal.classification,
        TerminalClassification::Failure { ref kind, .. } if kind == "NetworkError"
    ));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}
